import os
import sys
import asyncio
import functools
import json
import time
import logging
import base64
from typing import List, Dict, Any, Optional, Tuple

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.camera.stream import CameraStream
from app.hand_tracking.detector import HandDetector
from app.gestures.classifier import PalmAwareGestureClassifier, GestureState
from app.gestures.stabilizer import MultiHandStabilizer
from app.trajectory.engine import WordTrajectoryEngine
from app.recognition.trocr_engine import TrOCRHandwritingEngine
from app.services.movie_search import MovieSearchService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AirWrite-FastAPI")

app = FastAPI(
    title="AirWrite TV Search Backend",
    description="Computer vision, live MJPEG video stream, TrOCR recognition, and TV search service",
    version="1.0.0"
)

# Configure CORS for React frontend (Vite dev server http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_in_thread(func, *args):
    """Universal Python async thread executor compatible with all Python versions."""
    loop = asyncio.get_running_loop()
    if args:
        func = functools.partial(func, *args)
    return await loop.run_in_executor(None, func)

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                to_remove.append(connection)
        for conn in to_remove:
            self.disconnect(conn)

manager = ConnectionManager()

class CameraFeedManager:
    """Manages high-throughput zero-latency binary WebSocket video connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Camera binary WebSocket connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Camera binary WebSocket disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast_frame(self, frame_bytes: bytes):
        if not self.active_connections:
            return
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_bytes(frame_bytes)
            except Exception:
                to_remove.append(connection)
        for conn in to_remove:
            self.disconnect(conn)

camera_feed_manager = CameraFeedManager()

# Global CV Engine Control State
camera_stream: Optional[CameraStream] = None
cv_task: Optional[asyncio.Task] = None
is_cv_running = False
is_camera_active = False
is_recognizing = False
cv_lock = asyncio.Lock()

# Frame cache for MJPEG stream endpoint
latest_annotated_frame: Optional[np.ndarray] = None
latest_jpeg_bytes: Optional[bytes] = None
latest_frame_id: int = 0
last_rec_result: Optional[Dict[str, Any]] = None

# Pre-encoded standby frame (created once to save CPU)
standby_img = np.full((360, 640, 3), 20, dtype=np.uint8)
cv2.putText(standby_img, "CAMERA STANDBY", (210, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
cv2.putText(standby_img, "Click [ START CAMERA ] in UI", (170, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
_, standby_buf = cv2.imencode(".jpg", standby_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
STANDBY_FRAME_BYTES: bytes = (b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + standby_buf.tobytes() + b'\r\n')

# TrOCR Pretrained Engine Singleton
trocr_engine: Optional[TrOCRHandwritingEngine] = None
word_engine: Optional[WordTrajectoryEngine] = None

def draw_strokes_on_frame(frame: np.ndarray, strokes: List[List[Tuple[int, int]]], line_color=(255, 230, 0), thickness=4):
    """Renders multi-stroke visual handwriting on the preview frame with coordinate scaling."""
    h, w, _ = frame.shape
    scale_x = w / 1280.0
    scale_y = h / 720.0
    for stroke in strokes:
        if not stroke:
            continue
        if len(stroke) == 1:
            pt = (int(stroke[0][0] * scale_x), int(stroke[0][1] * scale_y))
            cv2.circle(frame, pt, thickness + 2, (0, 180, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, thickness, line_color, -1, cv2.LINE_AA)
            continue
        for i in range(1, len(stroke)):
            pt1 = (int(stroke[i - 1][0] * scale_x), int(stroke[i - 1][1] * scale_y))
            pt2 = (int(stroke[i][0] * scale_x), int(stroke[i][1] * scale_y))
            cv2.line(frame, pt1, pt2, (0, 180, 255), thickness + 4, cv2.LINE_AA)
            cv2.line(frame, pt1, pt2, line_color, thickness, cv2.LINE_AA)
    return frame

async def run_trocr_background_task(strokes_to_recognize: List[List[Tuple[int, int]]]):
    """Runs TrOCR recognition asynchronously in a threadpool so the live video stream never freezes."""
    global trocr_engine, is_recognizing, last_rec_result
    try:
        if trocr_engine is None:
            logger.info("Initializing TrOCR model in background for CONFIRM gesture...")
            trocr_engine = await run_in_thread(TrOCRHandwritingEngine, 0.38)

        logger.info(f"Running background TrOCR recognition on {len(strokes_to_recognize)} strokes...")
        t0 = time.time()
        rec_res = await run_in_thread(trocr_engine.recognize, strokes_to_recognize)
        latency_ms = int((time.time() - t0) * 1000)

        status = rec_res.status
        word = rec_res.text
        conf = rec_res.confidence
        cropped_img = rec_res.rendered_image

        img_b64 = ""
        if cropped_img is not None:
            _, buf = cv2.imencode(".png", cropped_img)
            img_b64 = base64.b64encode(buf).decode("utf-8")

        rec_msg = {
            "type": "recognition_result",
            "text": word,
            "confidence": conf,
            "status": status,
            "cropped_image_base64": img_b64,
            "strokes": strokes_to_recognize,
            "event": "CONFIRMED"
        }

        last_rec_result = {
            "status": status,
            "word": word,
            "confidence": conf,
            "latency_ms": latency_ms,
            "rendered_image": cropped_img
        }

        logger.info(f"TrOCR Finished ({latency_ms}ms) -> Word: '{word}' | Status: {status} | Conf: {conf*100:.1f}%")
        await manager.broadcast(rec_msg)

        final_state = "recognized" if status == "RECOGNIZED" else "ready"
        await manager.broadcast({
            "type": "status",
            "gesture_state": final_state,
            "strokes": strokes_to_recognize
        })
    except Exception as e:
        logger.error(f"Error during background TrOCR task: {e}", exc_info=True)
    finally:
        is_recognizing = False

async def cv_processing_loop():
    """Continuous non-blocking background loop processing camera stream, MediaPipe tracking & TrOCR recognition."""
    global camera_stream, is_cv_running, is_camera_active, trocr_engine, word_engine
    global latest_annotated_frame, latest_jpeg_bytes, latest_frame_id, is_recognizing, last_rec_result

    logger.info("Starting High-FPS CV Processing Loop...")
    
    detector = HandDetector(max_num_hands=2)
    stabilizer = MultiHandStabilizer()
    word_engine = WordTrajectoryEngine()

    is_cv_running = True
    is_camera_active = True
    last_broadcast_state = "ready"
    last_trajectory_broadcast_time = 0.0

    await manager.broadcast({
        "type": "camera_status",
        "camera_active": True,
        "gesture_state": "ready"
    })

    last_processed_frame_id = -1

    try:
        while is_cv_running and camera_stream and camera_stream.is_opened:
            # Check frame ID so we only process when the camera hardware produces a new frame
            latest_id = camera_stream.get_latest_frame_id()
            if latest_id == last_processed_frame_id:
                await asyncio.sleep(0.002)
                continue

            ret, frame = camera_stream.read_frame()
            if not ret or frame is None:
                await asyncio.sleep(0.002)
                continue
            last_processed_frame_id = latest_id

            h, w, _ = frame.shape
            
            # Run MediaPipe hand tracking off the main asyncio thread to keep event loop 100% responsive
            hands_data = await run_in_thread(detector.process, frame)
            active_keys = []

            active_write_point = None
            active_action_gesture = None
            detected_writing_hand = None

            for hand in hands_data:
                hand_label = hand["label"]
                hand_id = hand["hand_id"]
                hand_key = f"{hand_label}_{hand_id}"
                active_keys.append(hand_key)

                landmarks_norm = hand["landmarks_norm"]
                landmarks_pixel = hand["landmarks_pixel"]

                classification = PalmAwareGestureClassifier.classify_hand(landmarks_norm)
                raw_gesture = classification["gesture"]
                stable_gesture = stabilizer.update(hand_key, raw_gesture)
                is_write_active = (stable_gesture == GestureState.WRITE)

                # Canonical 1280x720 coordinate for virtual writing canvas
                canon_pt = None
                if len(landmarks_norm) > 8:
                    tip_norm = landmarks_norm[8]
                    canon_pt = (int(round(tip_norm[0] * 1280)), int(round(tip_norm[1] * 720)))

                if is_write_active and canon_pt:
                    detected_writing_hand = hand
                    active_write_point = canon_pt
                elif stable_gesture in (GestureState.CONFIRM, GestureState.CLEAR):
                    active_action_gesture = stable_gesture
                elif stable_gesture == GestureState.PEN_UP and not detected_writing_hand:
                    active_action_gesture = stable_gesture

                # Draw MediaPipe hand landmarks overlay
                frame = detector.draw_landmarks(frame, hand, is_writing_active=is_write_active)

                # Draw index fingertip point
                if is_write_active and len(landmarks_pixel) > 8:
                    idx_x, idx_y = landmarks_pixel[8]
                    cv2.circle(frame, (idx_x, idx_y), 16, (255, 255, 0), 2)
                    cv2.circle(frame, (idx_x, idx_y), 8, (255, 255, 0), -1)
                    cv2.putText(frame, f"WRITE TIP ({idx_x}, {idx_y})", (idx_x + 20, idx_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Render per-hand wrist status HUD
                wrist_x, wrist_y = landmarks_pixel[0]
                status_color = (0, 255, 0) if stable_gesture == GestureState.WRITE else \
                               (0, 200, 255) if stable_gesture == GestureState.CONFIRM else \
                               (0, 0, 255) if stable_gesture == GestureState.PEN_UP else \
                               (255, 100, 0) if stable_gesture == GestureState.CLEAR else (180, 180, 180)

                badge_text = f"{hand_label}: {stable_gesture}"
                cv2.rectangle(frame, (wrist_x - 10, wrist_y - 40), (wrist_x + 180, wrist_y - 10), (0, 0, 0), -1)
                cv2.putText(frame, badge_text, (wrist_x - 5, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            stabilizer.prune_missing(active_keys)

            now = time.time()

            # Determine engine input state with writing priority
            if active_action_gesture in (GestureState.CONFIRM, GestureState.CLEAR):
                current_input_state = active_action_gesture
            elif active_write_point is not None:
                current_input_state = GestureState.WRITE
            elif active_action_gesture == GestureState.PEN_UP:
                current_input_state = GestureState.PEN_UP
            else:
                current_input_state = GestureState.NEUTRAL

            # Handle gestures
            if current_input_state == GestureState.CONFIRM and not is_recognizing:
                drawable_strokes = word_engine.get_strokes_to_draw()
                if drawable_strokes and len(drawable_strokes) > 0:
                    is_recognizing = True
                    word_engine.update(GestureState.CONFIRM, active_write_point, processor=None, timestamp=now)
                    await manager.broadcast({
                        "type": "status",
                        "gesture_state": "processing",
                        "strokes": drawable_strokes
                    })
                    asyncio.create_task(run_trocr_background_task(drawable_strokes))
            elif current_input_state == GestureState.CLEAR:
                word_engine.clear()
                last_rec_result = None
                await manager.broadcast({"type": "clear", "gesture_state": "clear"})
                last_broadcast_state = "ready"
            elif not is_recognizing:
                word_engine.update(current_input_state, active_write_point, processor=None, timestamp=now)

            # Get drawable strokes & render on video frame
            drawable_strokes = word_engine.get_strokes_to_draw()
            if drawable_strokes:
                frame = draw_strokes_on_frame(frame, drawable_strokes)

            # Top HUD bar
            fps_val = int(camera_stream.fps) if camera_stream else 30
            cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
            hud_title = "AIRWRITE CV - LIVE BACKEND FEED"
            if is_recognizing:
                hud_title += " [TR-OCR RECOGNIZING...]"
            cv2.putText(frame, hud_title, (15, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if is_recognizing else (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps_val}", (w - 110, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            latest_annotated_frame = frame

            # Optimized MJPEG encoding (done once per tick, downsampled to stream resolution)
            stream_w = min(w, settings.STREAM_MAX_WIDTH)
            stream_h = int(h * (stream_w / float(w)))
            if stream_w < w:
                stream_frame = cv2.resize(frame, (stream_w, stream_h), interpolation=cv2.INTER_LINEAR)
            else:
                stream_frame = frame

            ret_enc, buf_enc = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_JPEG_QUALITY])
            if ret_enc:
                latest_jpeg_bytes = buf_enc.tobytes()
                latest_frame_id += 1
                # Instantaneous zero-latency push to binary WebSocket video subscribers
                if camera_feed_manager.active_connections:
                    await camera_feed_manager.broadcast_frame(latest_jpeg_bytes)

            # Map gesture state to string
            gesture_state_str = "processing" if is_recognizing else \
                                "writing" if active_write_point else \
                                "pen_up" if current_input_state == GestureState.PEN_UP else \
                                "confirm" if current_input_state == GestureState.CONFIRM else \
                                "clear" if current_input_state == GestureState.CLEAR else "ready"

            # Broadcast state changes immediately
            if gesture_state_str != last_broadcast_state and gesture_state_str != "processing":
                last_broadcast_state = gesture_state_str
                await manager.broadcast({
                    "type": "status",
                    "gesture_state": gesture_state_str,
                    "strokes": drawable_strokes
                })
                last_trajectory_broadcast_time = now
            # Low-latency trajectory updates (~30 FPS) to prevent WebSocket lag
            elif drawable_strokes and (now - last_trajectory_broadcast_time > 0.032):
                last_trajectory_broadcast_time = now
                await manager.broadcast({
                    "type": "trajectory_update",
                    "strokes": drawable_strokes
                })

            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        logger.info("CV Processing loop task cancelled.")
    except Exception as e:
        logger.error(f"Error in CV processing loop: {e}", exc_info=True)
    finally:
        detector.close()
        if camera_stream:
            await run_in_thread(camera_stream.release)
            camera_stream = None
        is_cv_running = False
        is_camera_active = False
        latest_annotated_frame = None
        latest_jpeg_bytes = None
        logger.info("CV Processing loop stopped and camera released.")
        await manager.broadcast({
            "type": "camera_status",
            "camera_active": False,
            "gesture_state": "ready"
        })

async def generate_mjpeg_frames():
    """
    Asynchronous generator yielding multipart JPEG frames for live camera streaming.
    Only yields brand-new frames, immediately skipping delayed frames to eliminate buffer backlog.
    Includes Content-Length header for immediate browser boundary parsing without latency.
    """
    last_sent_id = -1
    try:
        while True:
            if not is_camera_active or latest_jpeg_bytes is None:
                yield STANDBY_FRAME_BYTES
                await asyncio.sleep(0.05)
                continue

            if latest_frame_id != last_sent_id:
                last_sent_id = latest_frame_id
                frame_data = latest_jpeg_bytes
                chunk = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame_data)).encode('ascii') + b'\r\n\r\n' +
                    frame_data + b'\r\n'
                )
                yield chunk

            await asyncio.sleep(0.01)
    except (asyncio.CancelledError, GeneratorExit):
        pass

@app.on_event("startup")
async def startup_event():
    global trocr_engine
    logger.info("Pre-loading Microsoft TrOCR Engine in background thread...")
    try:
        trocr_engine = await run_in_thread(TrOCRHandwritingEngine, 0.38)
        logger.info("Microsoft TrOCR Engine pre-loaded successfully!")
    except Exception as e:
        logger.error(f"Error pre-loading TrOCR engine: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global is_cv_running, cv_task
    is_cv_running = False
    if cv_task:
        cv_task.cancel()

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AirWrite TV Search CV Engine",
        "phase": 6.5,
        "camera_active": is_camera_active,
        "trocr_loaded": trocr_engine is not None,
        "max_num_hands": settings.MAX_NUM_HANDS
    }

@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "airwrite-backend",
        "backend_connected": True,
        "camera_active": is_camera_active,
        "cv_running": is_cv_running,
        "trocr_loaded": trocr_engine is not None
    }

@app.get("/api/camera/stream")
async def video_feed():
    """Live MJPEG Video Stream of OpenCV Backend Feed with zero buffer headers."""
    return StreamingResponse(
        generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "close",
        }
    )

@app.get("/api/camera/frame")
async def single_camera_frame():
    """Instantaneous single JPEG frame endpoint with zero buffering."""
    if not is_camera_active or latest_jpeg_bytes is None:
        return Response(content=standby_buf.tobytes(), media_type="image/jpeg")
    return Response(
        content=latest_jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        }
    )

@app.websocket("/ws/camera")
async def ws_camera_feed(websocket: WebSocket):
    """Direct binary WebSocket endpoint streaming real-time JPEG frames with 0ms buffer lag."""
    await camera_feed_manager.connect(websocket)
    if latest_jpeg_bytes is not None and is_camera_active:
        try:
            await websocket.send_bytes(latest_jpeg_bytes)
        except Exception:
            pass
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        camera_feed_manager.disconnect(websocket)
    except Exception:
        camera_feed_manager.disconnect(websocket)

@app.post("/api/camera/start")
async def start_camera():
    global camera_stream, cv_task, is_cv_running, is_camera_active
    try:
        async with cv_lock:
            if is_camera_active and is_cv_running:
                return {"status": "already_running", "camera_active": True}

            logger.info("Starting Camera Stream via REST request...")
            camera_stream = CameraStream()
            
            # Run camera initialization in a background thread to prevent blocking the asyncio event loop!
            started = await run_in_thread(camera_stream.start)
            if not started:
                logger.error("Failed to start camera stream.")
                camera_stream = None
                return {"status": "error", "message": "Failed to open camera device", "camera_active": False}

            is_cv_running = True
            is_camera_active = True
            cv_task = asyncio.create_task(cv_processing_loop())
            return {"status": "started", "camera_active": True}
    except Exception as e:
        logger.error(f"Error in start_camera endpoint: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "camera_active": False}

@app.post("/api/camera/stop")
async def stop_camera():
    global is_cv_running, is_camera_active, camera_stream, cv_task
    try:
        async with cv_lock:
            if not is_camera_active and not is_cv_running:
                return {"status": "already_stopped", "camera_active": False}

            logger.info("Stopping Camera Stream via REST request...")
            is_cv_running = False
            is_camera_active = False

            if camera_stream:
                await run_in_thread(camera_stream.release)
                camera_stream = None

            if cv_task:
                cv_task.cancel()
                cv_task = None

            await manager.broadcast({
                "type": "camera_status",
                "camera_active": False,
                "gesture_state": "ready"
            })
            return {"status": "stopped", "camera_active": False}
    except Exception as e:
        logger.error(f"Error in stop_camera endpoint: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "camera_active": False}

@app.post("/api/session/clear")
async def clear_session():
    global word_engine
    if word_engine:
        word_engine.clear()
    await manager.broadcast({"type": "clear", "gesture_state": "ready"})
    return {"status": "cleared"}

@app.get("/api/search")
async def search_movies(q: str = Query(..., min_length=1, description="Movie/TV search query")):
    logger.info(f"Received search request for: '{q}'")
    results = await run_in_thread(MovieSearchService.search_movies, q)
    return {"query": q, "count": len(results), "movies": results}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await websocket.send_json({
        "type": "init",
        "backend_connected": True,
        "camera_active": is_camera_active,
        "gesture_state": "ready"
    })
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
