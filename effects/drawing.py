"""
Air Drawing effect
──────────────────
• Point with your index finger to draw.
• Make a fist to lift the 'pen' (stop drawing).
• Pinch (thumb + index close together) to clear the canvas.
"""

import cv2
import numpy as np
from .base import Effect


class AirDrawEffect(Effect):
    COLORS = [
        (255, 0, 0), # red
        (0, 255, 0), # green
        (0, 0, 255), # blue
        (0, 255, 255), # yellow
        (255, 0, 255), # magenta
        (255, 255, 0), # cyan
        (255, 255, 255), # white
        (0, 0, 0), # black
    ]

    def __init__(self):
        self._canvas = None # drawing layer
        self._prev_pts  = {} # last point per hand index
        self._color_idx = 0

    def apply(self, frame: np.ndarray, hands: list) -> np.ndarray:
        # initialize canvas if needed
        if self._canvas is None or self._canvas.shape != frame.shape:
            self._canvas = np.zeros_like(frame)

        for i, hand in enumerate(hands):
            gesture = hand.gesture

            if hand.is_pinching:
                # pinch = erase canvas
                self._canvas[:] = 0
                self._prev_pts[i] = None
                self._color_idx   = (self._color_idx + 1) % len(self.COLORS)
                continue

            if gesture in ("fist", "other"):
                self._prev_pts[i] = None
                continue

            # drawing mode
            pt = hand.index_tip
            color = self.COLORS[self._color_idx]

            prev = self._prev_pts.get(i)
            if prev is not None:
                cv2.line(self._canvas, prev, pt, color, thickness=5, lineType=cv2.LINE_AA)
            else:
                cv2.circle(self._canvas, pt, 3, color, -1)

            self._prev_pts[i] = pt

        # blend canvas into frame
        mask  = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame = cv2.add(frame, self._canvas)

        # instructions
        color_name = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "White", "Black"][self._color_idx]
        cv2.putText(frame, f"Color: {color_name}  |  Pinch=clear+next color  |  Fist=lift pen",
                    (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return frame