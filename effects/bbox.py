"""
BoundingBox Filter effect
─────────────────────────
• Hold both hands in frame — a live preview rectangle appears between your index fingertips.
• Pinch BOTH hands simultaneously to confirm and lock in the filter.
• Up to 5 filters stack on screen at once. Oldest is removed when limit is reached.
• Press 'f' to cycle through available filters before confirming.
• Press 'c' to clear all locked filters.
"""

import time
import cv2
import numpy as np
from .base import Effect

MAX_FILTERS  = 5
RECOMPUTE_EVERY = 3   # only rerun heavy filter every N frames (reduces lag)

FILTER_NAMES = ["Pixelate", "Glitch", "Thermal", "Neon Edges", "Blur"]

FILTER_COLORS = [
    (0,   200, 255),
    (180,  0,  255),
    (0,   255, 100),
    (255, 100,   0),
    (255, 255, 255),
]


# ── filter functions ──────────────────────────────────────────────────────────

def apply_pixelate(roi: np.ndarray, _tick: int) -> np.ndarray:
    h, w = roi.shape[:2]
    block = max(4, min(h, w) // 12)
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_glitch(roi: np.ndarray, tick: int) -> np.ndarray:
    out = roi.copy()
    h, w = roi.shape[:2]
    shift = max(4, w // 16)
    # Deterministic cycle based on tick — no random calls per frame
    phase = (tick // 15) % 3
    if phase == 0:
        out[:, shift:,  2] = roi[:, :-shift, 2]
        out[:, :-shift, 0] = roi[:, shift:,  0]
    elif phase == 1:
        out[:, :-shift, 2] = roi[:, shift:,  2]
        out[:, shift:,  0] = roi[:, :-shift, 0]
    else:
        out[:, shift:,  1] = roi[:, :-shift, 1]
        out[:, :-shift, 2] = roi[:, shift:,  2]
    # Single deterministic scanline tear per phase
    y = (h // 3) * (1 + phase % 2)
    out[y] = np.roll(out[y], shift * (1 - phase % 2 * 2), axis=0)
    return out


def apply_thermal(roi: np.ndarray, _tick: int) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def apply_neon(roi: np.ndarray, _tick: int) -> np.ndarray:
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    neon  = np.zeros_like(roi)
    neon[edges > 0] = (255, 255, 0)
    glow  = cv2.GaussianBlur(neon, (7, 7), 0)
    dark  = (roi * 0.15).astype(np.uint8)
    return cv2.add(dark, cv2.add(neon, glow))


def apply_blur(roi: np.ndarray, _tick: int) -> np.ndarray:
    return cv2.GaussianBlur(roi, (25, 25), 0)


FILTER_FNS = [apply_pixelate, apply_glitch, apply_thermal, apply_neon, apply_blur]


# ── main effect class ─────────────────────────────────────────────────────────

class BoundingBoxEffect(Effect):
    def __init__(self):
        self._locked: list[dict] = []
        self._filter_idx         = 0
        self._prev_pinching      = False
        self._last_confirm_time  = 0.0
        self._cooldown           = 1.5
        self._tick               = 0   # global frame counter for caching

    def cycle_filter(self):
        self._filter_idx = (self._filter_idx + 1) % len(FILTER_NAMES)

    def clear_filters(self):
        self._locked.clear()

    def apply(self, frame: np.ndarray, hands: list) -> np.ndarray:
        self._tick += 1

        # Render locked filters first (under preview)
        for f in self._locked:
            frame = self._render_filter(frame, f)

        if len(hands) < 2:
            self._prev_pinching = False
            self._draw_hud(frame, preview=False)
            return frame

        h0, h1   = hands[0], hands[1]
        pt0, pt1 = h0.index_tip, h1.index_tip
        both_pinching = h0.is_pinching and h1.is_pinching

        if both_pinching and not self._prev_pinching:
            self._confirm(pt0, pt1)

        self._prev_pinching = both_pinching

        if not both_pinching:
            color = FILTER_COLORS[self._filter_idx]
            x1 = min(pt0[0], pt1[0])
            y1 = min(pt0[1], pt1[1])
            x2 = max(pt0[0], pt1[0])
            y2 = max(pt0[1], pt1[1])
            self._draw_preview_rect(frame, x1, y1, x2, y2, color)
            cv2.circle(frame, pt0, 8, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt1, 8, color, -1, cv2.LINE_AA)

        self._draw_hud(frame, preview=True)
        return frame

    # ── internals ─────────────────────────────────────────────────────────────

    def _confirm(self, pt0, pt1):
        now = time.time()
        if now - self._last_confirm_time < self._cooldown:
            return
        self._last_confirm_time = now

        x1, y1 = min(pt0[0], pt1[0]), min(pt0[1], pt1[1])
        x2, y2 = max(pt0[0], pt1[0]), max(pt0[1], pt1[1])
        if (x2 - x1) < 30 or (y2 - y1) < 30:
            return

        if len(self._locked) >= MAX_FILTERS:
            self._locked.pop(0)

        self._locked.append(dict(
            x1=x1, y1=y1, x2=x2, y2=y2,
            filter_idx=self._filter_idx,
            cache=None,          # cached filtered ROI
            cache_tick=-999,     # tick when cache was last computed
        ))
        self._filter_idx = (self._filter_idx + 1) % len(FILTER_NAMES)

    def _render_filter(self, frame: np.ndarray, f: dict) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x1c = max(0, f["x1"]);  y1c = max(0, f["y1"])
        x2c = min(fw, f["x2"]); y2c = min(fh, f["y2"])
        if x2c <= x1c or y2c <= y1c:
            return frame

        # Only recompute the heavy filter every RECOMPUTE_EVERY frames
        if (f["cache"] is None or
                self._tick - f["cache_tick"] >= RECOMPUTE_EVERY):
            roi           = frame[y1c:y2c, x1c:x2c].copy()
            fn            = FILTER_FNS[f["filter_idx"]]
            f["cache"]    = fn(roi, self._tick)
            f["cache_tick"] = self._tick

        frame[y1c:y2c, x1c:x2c] = f["cache"]

        color = FILTER_COLORS[f["filter_idx"]]
        x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]

        # Outer colored border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        # Inner white separator — prevents color borders from blending together
        cv2.rectangle(frame, (x1+3, y1+3), (x2-3, y2-3), (255, 255, 255), 1, cv2.LINE_AA)

        # Label with dark backing so it's always readable
        label = FILTER_NAMES[f["filter_idx"]]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1+4, y1+4), (x1+tw+14, y1+th+12), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1+9, y1+th+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return frame

    def _draw_preview_rect(self, frame, x1, y1, x2, y2, color):
        L, T = 24, 3
        corners = [
            ((x1, y1), (x1+L, y1), (x1, y1+L)),
            ((x2, y1), (x2-L, y1), (x2, y1+L)),
            ((x1, y2), (x1+L, y2), (x1, y2-L)),
            ((x2, y2), (x2-L, y2), (x2, y2-L)),
        ]
        for apex, hpt, vpt in corners:
            cv2.line(frame, apex, hpt, color, T, cv2.LINE_AA)
            cv2.line(frame, apex, vpt, color, T, cv2.LINE_AA)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        label = f"[ {FILTER_NAMES[self._filter_idx]} ]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cx = (x1 + x2) // 2 - tw // 2
        cy = (y1 + y2) // 2 + th // 2
        cv2.putText(frame, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def _draw_hud(self, frame, preview: bool):
        fh, fw = frame.shape[:2]
        color  = FILTER_COLORS[self._filter_idx]
        name   = FILTER_NAMES[self._filter_idx]

        pill = f"Filter: {name}  (f=cycle)"
        (tw, _), _ = cv2.getTextSize(pill, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        px, py = fw - tw - 20, 36
        cv2.rectangle(frame, (px-8, py-18), (px+tw+4, py+6), (30, 30, 30), -1)
        cv2.putText(frame, pill, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        for i in range(MAX_FILTERS):
            filled    = i < len(self._locked)
            dot_color = FILTER_COLORS[self._locked[i]["filter_idx"]] if filled else (60, 60, 60)
            cx = fw - 20 - (MAX_FILTERS - 1 - i) * 22
            cy = fh - 20
            cv2.circle(frame, (cx, cy), 7, dot_color, -1 if filled else 1, cv2.LINE_AA)

        msg = ("Pinch both hands to lock  |  f=cycle  |  c=clear"
               if preview else "Show both hands to draw a box")
        cv2.putText(frame, msg, (10, fh - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)