import math
from typing import List, Tuple, Dict, Any

class GestureState:
    WRITE = "WRITE"        # ☝️ Index finger extended ONLY
    PEN_UP = "PEN_UP"      # 🖐️ Full open palm (all main fingers extended) — pauses stroke & preserves writing
    CONFIRM = "CONFIRM"    # ✌️ Two fingers (Index + Middle extended ONLY) — triggers TrOCR recognition
    CLEAR = "CLEAR"        # ✊ Fist (all main fingers folded) — resets canvas & session
    NEUTRAL = "NEUTRAL"    # Any other unassigned gesture

class PalmAwareGestureClassifier:
    """
    Palm-aware classifier for detecting hand finger states and mapping to mutually exclusive gestures:
      - WRITE (☝️): Index finger extended ONLY
      - PEN_UP (🖐️): Full open palm (3 or 4 fingers extended) — stops stroke & pauses recording
      - CONFIRM (✌️): Index + Middle extended ONLY — triggers TrOCR recognition
      - CLEAR (✊): Fist (all main fingers folded) — resets canvas & clears accumulated handwriting
      - NEUTRAL: Any unassigned gesture
    """
    
    @staticmethod
    def _dist_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Computes 3D Euclidean distance between two normalized points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    @classmethod
    def is_finger_extended(
        cls,
        landmarks: List[Tuple[float, float, float]],
        mcp_idx: int,
        pip_idx: int,
        dip_idx: int,
        tip_idx: int
    ) -> bool:
        """
        Determines whether a finger is extended using 3D palm-aware geometry.
        Checks:
          1. Distance from Wrist (0) to TIP vs Wrist to PIP
          2. Distance from MCP to TIP vs MCP to PIP
          3. Distance from MCP to TIP vs MCP to DIP (tip further out than DIP)
        """
        wrist = landmarks[0]
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        dip = landmarks[dip_idx]
        tip = landmarks[tip_idx]

        d_wrist_tip = cls._dist_3d(wrist, tip)
        d_wrist_pip = cls._dist_3d(wrist, pip)
        d_mcp_tip = cls._dist_3d(mcp, tip)
        d_mcp_pip = cls._dist_3d(mcp, pip)
        d_mcp_dip = cls._dist_3d(mcp, dip)

        # Extended condition: tip is significantly further from wrist and MCP than PIP and DIP are
        is_extended_wrist = d_wrist_tip > (d_wrist_pip * 1.05)
        is_extended_mcp = d_mcp_tip > (d_mcp_pip * 1.15)
        is_extended_dip = d_mcp_tip > (d_mcp_dip * 0.98)

        return is_extended_wrist and is_extended_mcp and is_extended_dip

    @classmethod
    def classify_hand(cls, landmarks_norm: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Classifies single hand gesture state based on 3D landmark array.
        Returns dictionary with:
          - gesture: str (WRITE | PEN_UP | CONFIRM | CLEAR | NEUTRAL)
          - finger_states: Dict of finger extension booleans
          - index_tip_norm: Tuple of (x, y, z) index tip coordinates if available
        """
        if not landmarks_norm or len(landmarks_norm) < 21:
            return {
                "gesture": GestureState.NEUTRAL,
                "finger_states": {},
                "index_tip_norm": None
            }

        wrist = landmarks_norm[0]

        # Check main fingers (Index, Middle, Ring, Pinky)
        index_ext = cls.is_finger_extended(landmarks_norm, 5, 6, 7, 8)
        middle_ext = cls.is_finger_extended(landmarks_norm, 9, 10, 11, 12)
        ring_ext = cls.is_finger_extended(landmarks_norm, 13, 14, 15, 16)
        pinky_ext = cls.is_finger_extended(landmarks_norm, 17, 18, 19, 20)

        d_wrist_index = cls._dist_3d(wrist, landmarks_norm[8])
        d_wrist_middle = cls._dist_3d(wrist, landmarks_norm[12])

        # Relative extension check: if index is extended and ring/pinky are folded,
        # verify whether middle finger is genuinely extended alongside index (CONFIRM)
        # or tucked in (WRITE)
        if index_ext and not ring_ext and not pinky_ext:
            if middle_ext:
                # In CONFIRM (peace sign), middle tip reaches similar distance as index tip
                if d_wrist_index > d_wrist_middle * 1.20:
                    # Middle finger is curled relative to index -> WRITE
                    middle_ext = False

        # Thumb extension check relative to Pinky MCP (17)
        thumb_tip = landmarks_norm[4]
        pinky_mcp = landmarks_norm[17]
        thumb_ip = landmarks_norm[3]
        d_thumb_pinky = cls._dist_3d(thumb_tip, pinky_mcp)
        d_ip_pinky = cls._dist_3d(thumb_ip, pinky_mcp)
        thumb_ext = d_thumb_pinky > (d_ip_pinky * 1.1)

        finger_states = {
            "thumb": thumb_ext,
            "index": index_ext,
            "middle": middle_ext,
            "ring": ring_ext,
            "pinky": pinky_ext
        }

        num_extended = sum([1 for ext in (index_ext, middle_ext, ring_ext, pinky_ext) if ext])

        # Mutually exclusive gesture classification rules:
        # 1. WRITE (☝️): Index finger extended ONLY
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            gesture = GestureState.WRITE
        # 2. PEN_UP (🖐️): Full open palm (at least 3 main fingers extended)
        elif num_extended >= 3:
            gesture = GestureState.PEN_UP
        # 3. CONFIRM (✌️): Index + Middle extended ONLY
        elif index_ext and middle_ext and not ring_ext and not pinky_ext:
            gesture = GestureState.CONFIRM
        # 4. CLEAR (✊): Fist (all main fingers folded)
        elif num_extended == 0:
            gesture = GestureState.CLEAR
        else:
            gesture = GestureState.NEUTRAL

        return {
            "gesture": gesture,
            "finger_states": finger_states,
            "index_tip_norm": landmarks_norm[8]
        }
