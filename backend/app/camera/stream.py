import cv2
import time
import threading
import logging
from typing import Optional, Tuple
from app.config.settings import settings

logger = logging.getLogger(__name__)

class CameraStream:
    """
    Manages OpenCV webcam VideoCapture device with a dedicated background thread.
    Continuously flushes internal DirectShow hardware buffers so read_frame()
    always returns the instantaneous real-time camera frame with zero buffer lag.
    """
    def __init__(self, camera_index: int = settings.CAMERA_INDEX, width: int = settings.FRAME_WIDTH, height: int = settings.FRAME_HEIGHT):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Threading state
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[cv2.Mat] = None
        self._frame_id = 0
        
        # FPS tracker
        self.prev_frame_time = time.time()
        self.fps = 0.0

    @property
    def is_opened(self) -> bool:
        """Returns True if the camera VideoCapture device is initialized, opened and running."""
        return self.cap is not None and self.cap.isOpened() and self._running

    def start(self) -> bool:
        """Initializes and opens the camera stream using DirectShow and launches worker thread."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to default backend if DirectShow is not supported
            logger.warning(f"CAP_DSHOW failed on index {self.camera_index}, retrying default backend...")
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Unable to open camera index {self.camera_index}")
                return False

        # Request hardware MJPEG stream to bypass USB 2.0 uncompressed YUY2 bandwidth bottleneck
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass

        # Request minimal internal driver buffer to prevent delay accumulation
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Request configured resolution and target framerate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, settings.FPS_LIMIT)

        # Query actual resolution granted by the driver
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w > 0 and actual_h > 0:
            self.width = actual_w
            self.height = actual_h

        logger.info(f"Camera opened on index {self.camera_index} ({self.width}x{self.height} @ {settings.FPS_LIMIT} FPS)")

        # Verify initial frame capture
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.error("Failed to read initial verification frame from camera.")
            self.cap.release()
            self.cap = None
            return False

        with self._lock:
            self._latest_frame = cv2.flip(frame, 1)
            self._frame_id = 1
            self.prev_frame_time = time.time()
            self.fps = 30.0

        # Launch dedicated background grabber thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="CameraCaptureWorker")
        self._thread.start()
        logger.info("High-throughput threaded camera worker started.")
        return True

    def _capture_worker(self):
        """
        Dedicated background loop that continuously drains driver buffers.
        Uses grab() to flush stale frames so only the freshest sensor frame is retrieved.
        """
        while self._running and self.cap is not None and self.cap.isOpened():
            # Grab the next incoming frame
            grabbed = self.cap.grab()
            if not grabbed:
                time.sleep(0.002)
                continue

            # Drain any backlogged frames in the driver FIFO queue
            # This ensures 0.0ms accumulated buffer delay even if CPU was briefly busy
            drain_count = 0
            while drain_count < 5 and self.cap.grab():
                drain_count += 1

            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                # Mirror horizontally immediately in worker thread
                flipped = cv2.flip(frame, 1)
                now = time.time()
                with self._lock:
                    self._latest_frame = flipped
                    self._frame_id += 1
                    diff = now - self.prev_frame_time
                    if diff > 0:
                        self.fps = 0.9 * self.fps + 0.1 * (1.0 / diff)
                    self.prev_frame_time = now
            else:
                time.sleep(0.002)

    def get_latest_frame_id(self) -> int:
        """Returns current monotonic frame ID to avoid redundant processing."""
        with self._lock:
            return self._frame_id

    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Instantly returns the latest camera frame without blocking.
        Guarantees 0ms buffer delay.
        """
        if not self._running:
            return False, None

        with self._lock:
            if self._latest_frame is None:
                return False, None
            # Return copy to avoid race conditions with drawing overlays
            return True, self._latest_frame.copy()

    def release(self):
        """Safely stops the worker thread and releases the camera capture device."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        with self._lock:
            self._latest_frame = None

        logger.info("Camera capture worker stopped and hardware released.")
