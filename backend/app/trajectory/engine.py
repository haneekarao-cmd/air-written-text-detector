import time
import math
from typing import List, Tuple, Optional, Dict, Any
from app.trajectory.tracker import FingertipSmoother, interpolate_points

PointWithTime = Tuple[int, int, float]  # (x_px, y_px, timestamp_ms)
Stroke = List[PointWithTime]

class WordSession:
    """Represents a continuous multi-stroke word writing session."""
    def __init__(self):
        self.strokes: List[Stroke] = []
        self.current_stroke: Optional[Stroke] = None
        self.start_time: float = time.time()
        self.last_update_time: float = time.time()
        self.state: str = "IDLE"  # IDLE | WRITING | PAUSED | CONFIRMED

    def add_point(self, pt_px: Tuple[int, int], timestamp: Optional[float] = None):
        """Adds a point to the current stroke with distance-based interpolation."""
        now = timestamp if timestamp is not None else time.time()
        now_ms = now * 1000.0
        self.last_update_time = now

        if self.current_stroke is None:
            self.current_stroke = []
            self.strokes.append(self.current_stroke)
            self.current_stroke.append((pt_px[0], pt_px[1], now_ms))
            return

        prev_pt = (self.current_stroke[-1][0], self.current_stroke[-1][1])
        if prev_pt == pt_px:
            return

        # Distance check for interpolation
        dist = math.hypot(pt_px[0] - prev_pt[0], pt_px[1] - prev_pt[1])
        if dist > 8.0 and dist <= 160.0:
            # Interpolate intermediate points
            interpolated = interpolate_points(prev_pt, pt_px, step_px=8.0)
            n_pts = len(interpolated)
            prev_time_ms = self.current_stroke[-1][2]
            time_step = (now_ms - prev_time_ms) / max(1, n_pts)
            for idx, ipt in enumerate(interpolated):
                if ipt != (self.current_stroke[-1][0], self.current_stroke[-1][1]):
                    t_pt = prev_time_ms + (idx + 1) * time_step
                    self.current_stroke.append((ipt[0], ipt[1], t_pt))
        else:
            self.current_stroke.append((pt_px[0], pt_px[1], now_ms))

    def pause_stroke(self):
        """Pauses recording (PEN_UP) without clearing strokes or finishing the session."""
        if self.current_stroke is not None:
            # Filter accidental micro-strokes (glitches under 3 points and < 5px path)
            if len(self.current_stroke) < 3:
                if len(self.current_stroke) == 1:
                    # Keep single deliberate dots
                    pass
                elif len(self.current_stroke) == 2:
                    p1 = self.current_stroke[0]
                    p2 = self.current_stroke[1]
                    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 4.0:
                        self.strokes.pop()
            self.current_stroke = None
        self.state = "PAUSED"

    def get_all_points(self) -> List[PointWithTime]:
        all_pts = []
        for s in self.strokes:
            all_pts.extend(s)
        return all_pts

    def get_total_point_count(self) -> int:
        return sum(len(s) for s in self.strokes)


class WordTrajectoryEngine:
    """
    Manages continuous multi-stroke word writing sessions.
    
    Guarantees:
      1. 🖐️ PEN_UP pauses current stroke, preserves completed strokes.
      2. ☝️ WRITE resumes/continues word writing across multiple strokes with 1 Euro filter.
      3. ✌️ CONFIRM freezes trajectory, triggers recognition exactly once, and retains strokes for view.
      4. ✊ CLEAR resets canvas and session.
      5. Tracking interruptions (up to 5 frames) do not destroy or break active strokes.
    """
    def __init__(self, grace_frames: int = 5):
        self.session: Optional[WordSession] = None
        self.last_recognized_word: Optional[str] = None
        self.last_confidence: float = 0.0
        self.last_features: Dict[str, Any] = {}
        self.last_strokes: List[List[Tuple[int, int]]] = []
        self.last_rendered_image = None
        self.last_status = "IDLE"
        
        # Continuity & smoothing
        self.smoother = FingertipSmoother(min_cutoff=1.2, beta=0.025, max_jump_dist=220.0)
        self.grace_frames = grace_frames
        self.missed_tracking_frames = 0
        self.last_known_point: Optional[Tuple[int, int]] = None

    def _extract_strokes(self, session: WordSession) -> List[List[Tuple[int, int]]]:
        drawable_strokes = []
        for stroke in session.strokes:
            drawable_strokes.append([(pt[0], pt[1]) for pt in stroke])
        return drawable_strokes

    def update(
        self,
        gesture_state: str,
        fingertip_px: Optional[Tuple[int, int]],
        processor: Optional[Any] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Updates word session state based on gesture input.
        Returns status report dict for UI & debug logging.
        """
        now = timestamp if timestamp is not None else time.time()
        result_event = {"event": "IDLE", "word": None, "confidence": 0.0}

        if gesture_state == "WRITE" and fingertip_px is not None:
            self.missed_tracking_frames = 0
            
            # Start fresh session if needed
            if self.session is None or self.session.state == "CONFIRMED":
                self.session = WordSession()
                self.last_strokes = []
                self.smoother.reset()

            # Smooth the coordinate with adaptive 1 Euro Filter
            smoothed_pt = self.smoother.filter_point(fingertip_px[0], fingertip_px[1], timestamp=now)
            self.last_known_point = smoothed_pt

            self.session.state = "WRITING"
            self.session.add_point(smoothed_pt, timestamp=now)
            self.last_strokes = self._extract_strokes(self.session)
            result_event["event"] = "WRITING"

        elif gesture_state == "CONFIRM":
            self.missed_tracking_frames = 0
            if self.session is not None and self.session.get_total_point_count() > 0:
                self.session.state = "CONFIRMED"
                drawable_strokes = self.get_strokes_to_draw()
                self.last_strokes = list(drawable_strokes)
                
                # Execute recognition engine if provided
                if processor is not None:
                    if hasattr(processor, "recognize"):
                        rec_res = processor.recognize(drawable_strokes)
                        self.last_recognized_word = rec_res.text
                        self.last_confidence = rec_res.confidence
                        self.last_status = rec_res.status
                        self.last_rendered_image = rec_res.rendered_image
                        self.last_debug_info = rec_res.debug_info
                    elif hasattr(processor, "process"):
                        proc_res = processor.process(self.session)
                        self.last_recognized_word = proc_res["word"]
                        self.last_confidence = proc_res["confidence"]
                        self.last_status = "RECOGNIZED"
                        self.last_rendered_image = None
                        self.last_debug_info = proc_res.get("features", {})
                else:
                    self.last_recognized_word = "WORD_CAPTURED"
                    self.last_confidence = 1.0
                    self.last_status = "RECOGNIZED"
                    self.last_rendered_image = None
                    self.last_debug_info = {}

                result_event = {
                    "event": "CONFIRMED",
                    "word": self.last_recognized_word,
                    "confidence": self.last_confidence,
                    "status": getattr(self, "last_status", "RECOGNIZED"),
                    "rendered_image": getattr(self, "last_rendered_image", None),
                    "debug_info": getattr(self, "last_debug_info", {})
                }

                # Reset active session cleanly after confirmation while retaining last_strokes
                self.session = None
                self.smoother.reset()
                self.last_known_point = None
            else:
                result_event["event"] = "CONFIRM_EMPTY"

        elif gesture_state == "CLEAR":
            self.clear()
            result_event["event"] = "CLEARED"

        elif gesture_state == "PEN_UP":
            # Explicit PEN_UP gesture: Immediately pause stroke and preserve completed writing
            self.missed_tracking_frames = 0
            self.smoother.reset()
            self.last_known_point = None
            if self.session is not None:
                self.session.pause_stroke()
                self.last_strokes = self._extract_strokes(self.session)
                result_event["event"] = "PAUSED"

        else:
            # NEUTRAL or temporary tracking absence: apply grace period
            self.missed_tracking_frames += 1
            if self.session is not None and self.session.state == "WRITING":
                if self.missed_tracking_frames <= self.grace_frames:
                    # Within grace period: maintain WRITING state to prevent breaking the stroke
                    result_event["event"] = "WRITING"
                else:
                    # Grace period expired: pause stroke gracefully
                    self.session.pause_stroke()
                    self.smoother.reset()
                    self.last_known_point = None
                    self.last_strokes = self._extract_strokes(self.session)
                    result_event["event"] = "PAUSED"
            elif self.session is not None:
                self.session.pause_stroke()
                self.last_strokes = self._extract_strokes(self.session)
                result_event["event"] = "PAUSED"

        return result_event

    def clear(self):
        """Completely resets active session and recognition memory."""
        self.session = None
        self.last_recognized_word = None
        self.last_confidence = 0.0
        self.last_features = {}
        self.last_strokes = []
        self.last_rendered_image = None
        self.missed_tracking_frames = 0
        self.last_known_point = None
        self.smoother.reset()

    def get_strokes_to_draw(self) -> List[List[Tuple[int, int]]]:
        """Returns simplified (x, y) stroke tuples for rendering."""
        if self.session is not None:
            return self._extract_strokes(self.session)
        return self.last_strokes
