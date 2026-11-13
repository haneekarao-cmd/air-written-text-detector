import time
from collections import deque
from typing import Dict, List
from app.config.settings import settings
from app.gestures.classifier import GestureState

class SingleHandStabilizer:
    """
    Stabilizes gesture predictions for a single hand using state-machine asymmetric hysteresis.
    
    Guarantees:
      - Quick response entering WRITE (2 consecutive frames).
      - Sticky WRITE state: ignores transient 1-3 frame NEUTRAL noise while writing.
      - Safe CLEAR debounce: requires 8 consecutive fist frames (~0.25s) to avoid accidental canvas wipes.
      - Robust CONFIRM debounce: requires 4 consecutive frames.
      - Stable PEN_UP transition: requires 3 consecutive frames.
    """
    def __init__(self, window_size: int = settings.DEBOUNCE_WINDOW_SIZE, threshold: int = settings.TRANSITION_THRESHOLD):
        self.window_size = window_size
        self.threshold = threshold
        self.current_state = GestureState.NEUTRAL
        
        # Consecutive frame counters for state transitions
        self._write_count = 0
        self._penup_count = 0
        self._confirm_count = 0
        self._clear_count = 0
        self._neutral_count = 0

    def update(self, raw_gesture: str) -> str:
        # Increment candidate counter, decrement others
        if raw_gesture == GestureState.WRITE:
            self._write_count += 1
            self._penup_count = 0
            self._confirm_count = 0
            self._clear_count = 0
            self._neutral_count = 0
        elif raw_gesture == GestureState.PEN_UP:
            self._penup_count += 1
            self._write_count = 0
            self._confirm_count = 0
            self._clear_count = 0
            self._neutral_count = 0
        elif raw_gesture == GestureState.CONFIRM:
            self._confirm_count += 1
            self._write_count = 0
            self._penup_count = 0
            self._clear_count = 0
            self._neutral_count = 0
        elif raw_gesture == GestureState.CLEAR:
            self._clear_count += 1
            self._write_count = 0
            self._penup_count = 0
            self._confirm_count = 0
            self._neutral_count = 0
        else: # NEUTRAL
            self._neutral_count += 1
            self._write_count = max(0, self._write_count - 1)
            self._penup_count = max(0, self._penup_count - 1)
            self._confirm_count = max(0, self._confirm_count - 1)
            self._clear_count = max(0, self._clear_count - 1)

        # State transition evaluation with asymmetric hysteresis
        if self.current_state == GestureState.WRITE:
            # While writing, stay in WRITE unless an intentional action is sustained
            if self._clear_count >= 8:
                self.current_state = GestureState.CLEAR
            elif self._confirm_count >= 4:
                self.current_state = GestureState.CONFIRM
            elif self._penup_count >= 3:
                self.current_state = GestureState.PEN_UP
            elif self._neutral_count > 5:
                # Only drop to NEUTRAL if finger has been ambiguous for > 5 frames
                self.current_state = GestureState.NEUTRAL
            else:
                # Keep WRITE active (absorbs transient pose wobbles)
                self.current_state = GestureState.WRITE

        elif self.current_state == GestureState.PEN_UP:
            if self._write_count >= 2:
                self.current_state = GestureState.WRITE
            elif self._confirm_count >= 4:
                self.current_state = GestureState.CONFIRM
            elif self._clear_count >= 8:
                self.current_state = GestureState.CLEAR
            elif self._neutral_count > 6:
                self.current_state = GestureState.NEUTRAL
            else:
                self.current_state = GestureState.PEN_UP

        elif self.current_state in (GestureState.CONFIRM, GestureState.CLEAR):
            # Transient triggers that reset back to ready/neutral once gesture released
            if self._write_count >= 2:
                self.current_state = GestureState.WRITE
            elif self._penup_count >= 3:
                self.current_state = GestureState.PEN_UP
            elif self._neutral_count >= 3:
                self.current_state = GestureState.NEUTRAL

        else: # NEUTRAL or initial state
            if self._write_count >= 2:
                self.current_state = GestureState.WRITE
            elif self._penup_count >= 3:
                self.current_state = GestureState.PEN_UP
            elif self._confirm_count >= 4:
                self.current_state = GestureState.CONFIRM
            elif self._clear_count >= 8:
                self.current_state = GestureState.CLEAR

        return self.current_state

    def reset(self):
        self.current_state = GestureState.NEUTRAL
        self._write_count = 0
        self._penup_count = 0
        self._confirm_count = 0
        self._clear_count = 0
        self._neutral_count = 0


class MultiHandStabilizer:
    """Manages independent gesture stabilizers per hand key with decay persistence."""
    def __init__(self, window_size: int = settings.DEBOUNCE_WINDOW_SIZE, threshold: int = settings.TRANSITION_THRESHOLD):
        self.window_size = window_size
        self.threshold = threshold
        self.stabilizers: Dict[str, SingleHandStabilizer] = {}
        self.last_seen: Dict[str, float] = {}

    def get_stabilizer(self, hand_key: str) -> SingleHandStabilizer:
        if hand_key not in self.stabilizers:
            self.stabilizers[hand_key] = SingleHandStabilizer(self.window_size, self.threshold)
        return self.stabilizers[hand_key]

    def update(self, hand_key: str, raw_gesture: str) -> str:
        self.last_seen[hand_key] = time.time()
        stabilizer = self.get_stabilizer(hand_key)
        return stabilizer.update(raw_gesture)

    def prune_missing(self, active_keys: List[str]):
        """
        Removes stabilizers only after hand has been absent for at least 1.5 seconds.
        Prevents single-frame dropouts from erasing tracking history.
        """
        now = time.time()
        for key in active_keys:
            self.last_seen[key] = now

        for key in list(self.stabilizers.keys()):
            if key not in active_keys:
                if now - self.last_seen.get(key, 0) > 1.5:
                    del self.stabilizers[key]
                    if key in self.last_seen:
                        del self.last_seen[key]
