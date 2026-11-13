import cv2
import math
import time
from typing import List, Tuple, Optional

Point = Tuple[int, int]
Stroke = List[Point]

class LowPassFilter:
    """1st order low-pass filter for smoothing time-series signals."""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self):
        self.s = None


class OneEuroFilter:
    """
    Adaptive 1 Euro Filter (Casiez et al., CHI 2012).
    Dynamically balances jitter elimination at low speed with zero phase lag at high speed.
    """
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value: float, timestamp: float) -> float:
        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter(value, 1.0)

        dt = max(1e-4, timestamp - self.last_time)
        self.last_time = timestamp

        prev_val = self.x_filter.s if self.x_filter.s is not None else value
        dx = (value - prev_val) / dt
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(value, self._alpha(cutoff, dt))

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class FingertipSmoother:
    """
    2D coordinate smoother with adaptive 1 Euro filtering and jump-anomaly rejection.
    Ensures fingertip tracks smoothly like a real pen on paper.
    """
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.025, max_jump_dist: float = 220.0):
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.max_jump_dist = max_jump_dist
        self.last_output: Optional[Point] = None
        self.last_time: Optional[float] = None

    def filter_point(self, x: float, y: float, timestamp: Optional[float] = None) -> Point:
        if timestamp is None:
            timestamp = time.time()

        # Reject sudden tracking teleports (> max_jump_dist within 100ms)
        if self.last_output is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt < 0.12:
                dist = math.hypot(x - self.last_output[0], y - self.last_output[1])
                if dist > self.max_jump_dist:
                    # Anomaly clamp: step at maximum allowed velocity
                    scale = self.max_jump_dist / dist
                    x = self.last_output[0] + (x - self.last_output[0]) * scale
                    y = self.last_output[1] + (y - self.last_output[1]) * scale

        fx = self.filter_x.filter(x, timestamp)
        fy = self.filter_y.filter(y, timestamp)
        out_pt = (int(round(fx)), int(round(fy)))
        self.last_output = out_pt
        self.last_time = timestamp
        return out_pt

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()
        self.last_output = None
        self.last_time = None


def interpolate_points(pt1: Point, pt2: Point, step_px: float = 8.0) -> List[Point]:
    """
    Generates densely interpolated intermediate points between two coordinates
    if distance exceeds step_px to prevent disconnected segments during fast movement.
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = math.hypot(dx, dy)
    if dist <= step_px:
        return [pt2]

    num_steps = max(1, int(dist / step_px))
    interpolated = []
    for step in range(1, num_steps + 1):
        t = step / float(num_steps)
        ix = int(round(pt1[0] + t * dx))
        iy = int(round(pt1[1] + t * dy))
        interpolated.append((ix, iy))
    return interpolated


class TrajectoryTracker:
    """
    Tracks and stores multi-stroke air-writing trajectories.
    Allows independent strokes created across multiple WRITE sessions,
    retaining visible strokes during PEN_UP, and clearing on CONFIRM or CLEAR.
    """
    def __init__(self, max_strokes: int = 50):
        self.strokes: List[Stroke] = []
        self.current_stroke: Optional[Stroke] = None
        self.max_strokes = max_strokes
        self.smoother = FingertipSmoother()

    def update(self, gesture_state: str, fingertip_px: Optional[Point]):
        """
        Updates stroke records based on current gesture state:
          - WRITE: Appends smoothed fingertip pixel coordinates to active stroke.
          - PEN_UP / NEUTRAL: Finalizes active stroke (pen up).
          - CONFIRM / CLEAR: Clears all strokes from canvas.
        """
        if gesture_state == "WRITE" and fingertip_px is not None:
            smoothed = self.smoother.filter_point(fingertip_px[0], fingertip_px[1])
            if self.current_stroke is None:
                self.current_stroke = [smoothed]
                self.strokes.append(self.current_stroke)
            else:
                prev_pt = self.current_stroke[-1]
                dense_pts = interpolate_points(prev_pt, smoothed)
                for pt in dense_pts:
                    if pt != self.current_stroke[-1]:
                        self.current_stroke.append(pt)

        elif gesture_state in ("CONFIRM", "CLEAR"):
            self.clear()
        else:
            # PEN_UP or NEUTRAL state: End current stroke segment
            self.current_stroke = None
            self.smoother.reset()

    def clear(self):
        """Clears all captured strokes."""
        self.strokes.clear()
        self.current_stroke = None
        self.smoother.reset()

    def get_point_count(self) -> int:
        """Returns total number of points captured across all strokes."""
        return sum(len(s) for s in self.strokes)

    def draw_trajectory(self, frame: cv2.Mat, line_color: Tuple[int, int, int] = (255, 230, 0), thickness: int = 5) -> cv2.Mat:
        """
        Renders all strokes onto frame with smooth anti-aliased lines and glow effect.
        """
        for stroke in self.strokes:
            if len(stroke) < 2:
                if len(stroke) == 1:
                    cv2.circle(frame, stroke[0], thickness, line_color, -1)
                continue

            for i in range(1, len(stroke)):
                pt1 = stroke[i - 1]
                pt2 = stroke[i]
                # Outer glow
                cv2.line(frame, pt1, pt2, (0, 180, 255), thickness + 4, cv2.LINE_AA)
                # Core bright line
                cv2.line(frame, pt1, pt2, line_color, thickness, cv2.LINE_AA)

        return frame
