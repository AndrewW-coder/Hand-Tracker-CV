"""
detector.py – MediaPipe Tasks API version (Python 3.13 compatible)

Download the model first:
  curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

from dataclasses import dataclass, field
from turtle import up
from typing import List
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# connects the hand landmarks to other ones
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), # thumb
    (0,5),(5,6),(6,7),(7,8), # index
    (0,9),(9,10),(10,11),(11,12), # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17), # palm
]


@dataclass
class Hand:
    landmarks:  List[tuple] # list of coords
    handedness: str # left or right

    def tip(self, landmark_id: int) -> tuple:
        return self.landmarks[landmark_id]

    @property
    def index_tip(self) -> tuple:
        return self.tip(INDEX_TIP)

    @property
    def pinch_distance(self) -> float:
        tx, ty = self.tip(THUMB_TIP)
        ix, iy = self.tip(INDEX_TIP)
        return ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5

    @property
    def is_pinching(self) -> bool:
        return self.pinch_distance < 40

    # returns list of extended fingers
    @property
    def fingers_up(self) -> List[bool]:
        lm = self.landmarks
        up = []
        palm_center_x = (lm[0][0] + lm[5][0] + lm[17][0]) // 3
        palm_center_y = (lm[0][1] + lm[5][1] + lm[17][1]) // 3
        thumb_tip_dist = (
            (lm[4][0] - palm_center_x) ** 2 +
            (lm[4][1] - palm_center_y) ** 2
        ) ** 0.5
        index_mcp_dist = (
            (lm[5][0] - palm_center_x) ** 2 +
            (lm[5][1] - palm_center_y) ** 2
        ) ** 0.5
        up.append(thumb_tip_dist > index_mcp_dist * 1.1)

        for tip, pip, mcp in [(8,7,6), (12,11,10), (16,15,14), (20,19,18)]:
            up.append(lm[tip][1] < lm[pip][1] and lm[tip][1] < lm[mcp][1])

        return up

    @property
    def gesture(self) -> str:
        up = self.fingers_up
        n  = sum(up)
        if n == 0: return "fist"
        if n == 5: return "open"
        if up[1] and not any(up[2:]): return "point"
        if up[1] and up[2] and not any(up[3:]): return "peace"
        if up[0] and up[1] and not any(up[2:]): return "gun"
        return "other"


class HandDetector:
    # initializes detector with params
    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}\n"
                "Download it with:\n"
                "  curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_ts   = 0   # monotonic ms counter for VIDEO mode

    # returns hand param and draws
    def find_hands(self, frame: np.ndarray, draw_landmarks: bool = True) -> List[Hand]:
        """Process a BGR frame, return Hand objects, optionally draw skeleton."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33   # ~30 fps; Tasks VIDEO mode needs monotonic timestamps
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        hands = []
        if not result.hand_landmarks:
            return hands

        h, w = frame.shape[:2]

        for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            hand = Hand(
                landmarks=landmarks,
                handedness=handedness_list[0].display_name,
            )
            hands.append(hand)

            if draw_landmarks:
                self._draw_hand(frame, landmarks)

        return hands

    # helper draw func
    def _draw_hand(self, frame: np.ndarray, landmarks: List[tuple]):
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, landmarks[a], landmarks[b], (255, 255, 255), 2, cv2.LINE_AA)
        for x, y in landmarks:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)