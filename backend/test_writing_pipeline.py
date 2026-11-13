import os
import sys
import time
import math

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.gestures.classifier import PalmAwareGestureClassifier, GestureState
from app.gestures.stabilizer import SingleHandStabilizer, MultiHandStabilizer
from app.trajectory.tracker import OneEuroFilter, FingertipSmoother, interpolate_points
from app.trajectory.engine import WordSession, WordTrajectoryEngine
from app.recognition.canvas import VirtualHandwritingCanvas

def test_one_euro_filter():
    print("\n--- Test 1: One Euro Filter & Fingertip Smoother ---")
    smoother = FingertipSmoother(min_cutoff=1.2, beta=0.025, max_jump_dist=220.0)
    
    # 1. Steady position with jitter: smoothed should have greatly reduced variance
    t = 0.0
    raw_points = [(100 + (2 if i % 2 == 0 else -2), 200 + (2 if i % 2 == 0 else -2)) for i in range(20)]
    smoothed_points = []
    for x, y in raw_points:
        s = smoother.filter_point(x, y, timestamp=t)
        smoothed_points.append(s)
        t += 0.033
    
    # Verify smoothed deviation is smaller than raw deviation
    raw_xs = [p[0] for p in raw_points[5:]]
    smooth_xs = [p[0] for p in smoothed_points[5:]]
    raw_var = max(raw_xs) - min(raw_xs)
    smooth_var = max(smooth_xs) - min(smooth_xs)
    assert smooth_var < raw_var, f"Smoothed variance {smooth_var} should be < raw variance {raw_var}"
    print(f"  [PASS] Jitter reduced: raw spread = {raw_var}px, smoothed spread = {smooth_var}px")

    # 2. Jump anomaly rejection: massive 300px jump in 33ms clamped to max_jump_dist
    p_before = smoothed_points[-1]
    p_jump = smoother.filter_point(p_before[0] + 350, p_before[1] + 350, timestamp=t)
    dist = math.hypot(p_jump[0] - p_before[0], p_jump[1] - p_before[1])
    assert dist <= 230.0, f"Jump should be clamped, got step distance {dist}"
    print(f"  [PASS] Anomaly jump clamped: requested 495px step, clamped to {dist:.1f}px")

def test_interpolation():
    print("\n--- Test 2: Distance-Based Point Interpolation ---")
    p1 = (100, 100)
    p2 = (140, 130) # dist = 50px
    pts = interpolate_points(p1, p2, step_px=8.0)
    assert len(pts) >= 5, f"Expected at least 5 interpolated points, got {len(pts)}"
    # Verify final point equals destination
    assert pts[-1] == p2, f"Final interpolated point should match target {p2}, got {pts[-1]}"
    print(f"  [PASS] Interpolated 50px span into {len(pts)} smooth sub-points")

def test_gesture_classifier():
    print("\n--- Test 3: Gesture Classifier & Geometry ---")
    # Generate synthetic 21-landmark arrays
    # 0 is wrist at (0.5, 0.8, 0.0)
    wrist = (0.5, 0.8, 0.0)

    # Helper to create hand landmarks:
    def make_hand(index_extended=True, middle_extended=False, ring_extended=False, pinky_extended=False):
        lms = [wrist] * 21
        # Wrist at 0
        lms[0] = (0.5, 0.8, 0.0)

        # Index: MCP(5), PIP(6), DIP(7), TIP(8)
        lms[5] = (0.45, 0.6, 0.0)
        lms[6] = (0.44, 0.5, 0.0)
        lms[7] = (0.43, 0.4, 0.0)
        lms[8] = (0.42, 0.2, 0.0) if index_extended else (0.45, 0.65, 0.0)

        # Middle: MCP(9), PIP(10), DIP(11), TIP(12)
        lms[9] = (0.5, 0.6, 0.0)
        lms[10] = (0.5, 0.5, 0.0)
        lms[11] = (0.5, 0.4, 0.0)
        lms[12] = (0.5, 0.2, 0.0) if middle_extended else (0.5, 0.65, 0.0)

        # Ring: MCP(13), PIP(14), DIP(15), TIP(16)
        lms[13] = (0.55, 0.6, 0.0)
        lms[14] = (0.55, 0.5, 0.0)
        lms[15] = (0.55, 0.4, 0.0)
        lms[16] = (0.55, 0.2, 0.0) if ring_extended else (0.55, 0.65, 0.0)

        # Pinky: MCP(17), PIP(18), DIP(19), TIP(20)
        lms[17] = (0.6, 0.6, 0.0)
        lms[18] = (0.6, 0.5, 0.0)
        lms[19] = (0.6, 0.4, 0.0)
        lms[20] = (0.6, 0.2, 0.0) if pinky_extended else (0.6, 0.65, 0.0)

        return lms

    # Test WRITE (☝️ only index extended)
    write_hand = make_hand(index_extended=True, middle_extended=False, ring_extended=False, pinky_extended=False)
    res_write = PalmAwareGestureClassifier.classify_hand(write_hand)
    assert res_write["gesture"] == GestureState.WRITE, f"Expected WRITE, got {res_write['gesture']}"
    print("  [PASS] Index finger only classified as WRITE")

    # Test CONFIRM (✌️ index + middle extended)
    confirm_hand = make_hand(index_extended=True, middle_extended=True, ring_extended=False, pinky_extended=False)
    res_confirm = PalmAwareGestureClassifier.classify_hand(confirm_hand)
    assert res_confirm["gesture"] == GestureState.CONFIRM, f"Expected CONFIRM, got {res_confirm['gesture']}"
    print("  [PASS] Index + Middle fingers classified as CONFIRM")

    # Test PEN_UP (🖐️ all fingers extended)
    penup_hand = make_hand(index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=True)
    res_penup = PalmAwareGestureClassifier.classify_hand(penup_hand)
    assert res_penup["gesture"] == GestureState.PEN_UP, f"Expected PEN_UP, got {res_penup['gesture']}"
    print("  [PASS] Open palm classified as PEN_UP")

    # Test CLEAR (✊ all fingers folded)
    clear_hand = make_hand(index_extended=False, middle_extended=False, ring_extended=False, pinky_extended=False)
    res_clear = PalmAwareGestureClassifier.classify_hand(clear_hand)
    assert res_clear["gesture"] == GestureState.CLEAR, f"Expected CLEAR, got {res_clear['gesture']}"
    print("  [PASS] Fist classified as CLEAR")

def test_stabilizer_hysteresis():
    print("\n--- Test 4: Gesture Stabilizer Hysteresis & Debounce ---")
    stab = SingleHandStabilizer()

    # Fast 2-frame entry into WRITE
    assert stab.update(GestureState.WRITE) == GestureState.NEUTRAL
    assert stab.update(GestureState.WRITE) == GestureState.WRITE
    print("  [PASS] Fast 2-frame entry into WRITE verified")

    # Sticky WRITE: 1-frame transient NEUTRAL does not break WRITE
    assert stab.update(GestureState.NEUTRAL) == GestureState.WRITE
    assert stab.update(GestureState.NEUTRAL) == GestureState.WRITE
    print("  [PASS] Transient NEUTRAL ignored; WRITE maintained")

    # Resume WRITE
    assert stab.update(GestureState.WRITE) == GestureState.WRITE

    # Accidental 1-frame fist does NOT trigger CLEAR
    assert stab.update(GestureState.CLEAR) == GestureState.WRITE
    assert stab.update(GestureState.CLEAR) == GestureState.WRITE
    assert stab.update(GestureState.WRITE) == GestureState.WRITE
    print("  [PASS] Accidental fist frame rejected without triggering CLEAR")

    # Sustained 8-frame fist triggers CLEAR
    for _ in range(8):
        last_state = stab.update(GestureState.CLEAR)
    assert last_state == GestureState.CLEAR
    print("  [PASS] Sustained 8-frame fist correctly triggers CLEAR")

def test_word_trajectory_continuity():
    print("\n--- Test 5: WordTrajectoryEngine Continuity & Grace Period ---")
    engine = WordTrajectoryEngine(grace_frames=5)

    # Write first stroke (e.g. letter 'L' down)
    engine.update("WRITE", (200, 200))
    engine.update("WRITE", (200, 250))
    engine.update("WRITE", (200, 300))
    assert engine.session.state == "WRITING"
    assert len(engine.session.strokes) == 1
    p_count_before = len(engine.session.strokes[0])
    print(f"  [PASS] Initial stroke created with {p_count_before} points")

    # MediaPipe dropout for 3 frames (grace period)
    engine.update("NEUTRAL", None)
    engine.update("NEUTRAL", None)
    engine.update("NEUTRAL", None)
    assert engine.session.state == "WRITING"
    assert len(engine.session.strokes) == 1, "Stroke should NOT be split during grace period"
    print("  [PASS] 3-frame tracking dropout handled inside grace period without breaking stroke")

    # Hand reappears in WRITE: should continue SAME stroke
    engine.update("WRITE", (250, 300))
    engine.update("WRITE", (300, 300))
    assert len(engine.session.strokes) == 1, "Stroke should still be single continuous stroke"
    assert len(engine.session.strokes[0]) > p_count_before
    print("  [PASS] Hand re-detection seamlessly continued the same stroke")

    # Open palm (🖐️ PEN_UP) pauses stroke and preserves it
    engine.update("PEN_UP", None)
    assert engine.session.state == "PAUSED"
    assert engine.session.current_stroke is None
    strokes_after_pause = engine.get_strokes_to_draw()
    assert len(strokes_after_pause) == 1
    print("  [PASS] PEN_UP paused stroke and preserved completed stroke on canvas")

    # Write second stroke (cross stroke)
    engine.update("WRITE", (180, 220))
    engine.update("WRITE", (240, 220))
    assert len(engine.session.strokes) == 2, "Multi-stroke word should now have 2 strokes"
    print("  [PASS] Second stroke created, forming complete multi-stroke word")

    # Confirm word (✌️ CONFIRM)
    ev = engine.update("CONFIRM", None)
    assert ev["event"] == "CONFIRMED"
    # Verify strokes remain drawable for display after confirmation
    strokes_preserved = engine.get_strokes_to_draw()
    assert len(strokes_preserved) == 2, "Strokes should remain visible after confirmation"
    print("  [PASS] CONFIRM froze word and preserved strokes for display")

    # Clear word (✊ CLEAR)
    engine.update("CLEAR", None)
    assert len(engine.get_strokes_to_draw()) == 0
    print("  [PASS] CLEAR erased canvas completely")

if __name__ == "__main__":
    print("=================================================================")
    print("AIRWRITE REAL-TIME WRITING PIPELINE VERIFICATION SUITE")
    print("=================================================================")
    test_one_euro_filter()
    test_interpolation()
    test_gesture_classifier()
    test_stabilizer_hysteresis()
    test_word_trajectory_continuity()
    print("\n=================================================================")
    print("ALL 5 CRITICAL WRITING PIPELINE TESTS PASSED!")
    print("=================================================================")

