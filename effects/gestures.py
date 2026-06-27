"""
Gesture Visualiser effect
─────────────────────────
Displays the detected gesture name and finger states in real time.
Great as a debugging layer while you build your own gesture-driven logic.
"""

import cv2
import numpy as np
from .base import Effect

GESTURE_COLORS = {
    "fist":  (50,  50, 255),
    "open":  (50, 255, 50),
    "point": (255, 200, 0),
    "peace": (200, 0, 255),
    "gun":   (0, 200, 255),
    "other": (180, 180, 180),
}

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


class GestureEffect(Effect):
    def apply(self, frame: np.ndarray, hands: list) -> np.ndarray:
        for i, hand in enumerate(hands):
            gesture = hand.gesture
            color   = GESTURE_COLORS.get(gesture, (255, 255, 255))

            # gesture label
            wx, wy = hand.tip(0)          # landmark 0 = wrist
            cv2.putText(frame, gesture.upper(), (wx - 40, wy + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3, cv2.LINE_AA)

            # finger dots
            TIP_IDS = [4, 8, 12, 16, 20]
            for is_up, tip_id, name in zip(hand.fingers_up, TIP_IDS, FINGER_NAMES):
                tx, ty = hand.tip(tip_id)
                dot_color = (0, 255, 100) if is_up else (0, 60, 200)
                cv2.circle(frame, (tx, ty - 20), 8, dot_color, -1, cv2.LINE_AA)

            # pinch dist
            pd = hand.pinch_distance
            bar_w = min(int(pd * 2), 200)
            bar_x, bar_y = 20 + i * 220, frame.shape[0] - 80
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + 18), (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18),
                          (0, 200, 255) if hand.is_pinching else (100, 100, 255), -1)
            cv2.putText(frame, f"Pinch {pd:.0f}px", (bar_x, bar_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if not hands:
            h, w = frame.shape[:2]
            cv2.putText(frame, "No hands detected", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        return frame