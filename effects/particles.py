"""
Particle Trail effect
─────────────────────
Colourful sparks follow your fingertips.
Open hand = more particles.  Fist = no particles.
"""

import random
import cv2
import numpy as np
from .base import Effect


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "radius")

    def __init__(self, x: int, y: int, color: tuple):
        self.x       = float(x)
        self.y       = float(y)
        self.vx      = random.uniform(-4, 4)
        self.vy      = random.uniform(-6, -1)   # mostly upward
        self.max_life = random.randint(20, 50)
        self.life    = self.max_life
        self.color   = color
        self.radius  = random.randint(2, 5)

    def update(self):
        self.vy   += 0.25          # gravity
        self.x    += self.vx
        self.y    += self.vy
        self.life -= 1

    @property
    def alive(self) -> bool:
        return self.life > 0

    @property
    def alpha(self) -> float:
        return self.life / self.max_life


TIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky

PALETTES = [
    [(0, 200, 255), (0, 255, 200), (0, 150, 255)],   # cyan / teal
    [(255, 100, 0), (255, 200, 0), (255, 50, 50)],    # fire
    [(200, 0, 255), (255, 0, 150), (100, 0, 255)],    # purple / pink
]


class ParticleEffect(Effect):
    def __init__(self):
        self._particles: list[Particle] = []
        self._palette_idx = 0

    def apply(self, frame: np.ndarray, hands: list) -> np.ndarray:
        overlay = frame.copy()

        for hand in hands:
            up    = hand.fingers_up
            n_up  = sum(up)

            if n_up == 0:
                # Fist: cycle palette
                self._palette_idx = (self._palette_idx + 1) % len(PALETTES)
                continue

            palette = PALETTES[self._palette_idx]

            # Spawn particles only at extended fingertips
            for idx, (tip_id, is_up) in enumerate(zip(TIP_IDS, up)):
                if not is_up:
                    continue
                x, y = hand.tip(tip_id)
                color = palette[idx % len(palette)]
                for _ in range(3):          # 3 particles per tip per frame
                    self._particles.append(Particle(x, y, color))

        # Update & draw all particles
        alive = []
        for p in self._particles:
            p.update()
            if p.alive:
                alive.append(p)
                b, g, r = p.color
                # Fade colour toward black
                faded = (int(b * p.alpha), int(g * p.alpha), int(r * p.alpha))
                cv2.circle(overlay, (int(p.x), int(p.y)), p.radius, faded, -1,
                           lineType=cv2.LINE_AA)

        self._particles = alive

        # Soft blend so particles glow slightly
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        palette_names = ["Cyan", "Fire", "Purple"]
        cv2.putText(frame, f"Palette: {palette_names[self._palette_idx]}  |  Fist=cycle palette",
                    (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return frame