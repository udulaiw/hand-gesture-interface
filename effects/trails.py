"""Fingertip trails.

Two independent layers share the same tapered-ribbon renderer:

  trail   a short wake behind the index tip, length driven by speed. It exists
          to make fast motion legible, so it dies in a fraction of a second.
  stroke  the two-finger 'draw' mode. Strokes persist for seconds and are
          cleared by a wave or by R.
"""
from collections import deque

import numpy as np

import config
from tracking.gestures import PEACE, PINCH, POINT
from utils.smoothing import clamp

TRAIL_TTL = 0.42
STROKE_TTL = 4.5


def _ribbon(ov, points, ages, ttl, color, gain, max_width):
    n = len(points)
    if n < 2:
        return
    for i in range(n - 1):
        a = 1.0 - ages[i] / ttl
        if a <= 0.02:
            continue
        # Taper toward the tail as well as fading.
        head = (i + 1) / n
        alpha = (a ** 1.4) * gain * (0.35 + 0.65 * head)
        wdt = max(1, int(round(max_width * a * head)))
        ov.line(points[i], points[i + 1], color, alpha, wdt)


class TrailSystem:
    def __init__(self, n_hands):
        self._trail = [deque(maxlen=config.TRAIL_LENGTH) for _ in range(n_hands)]
        self._strokes = [[] for _ in range(n_hands)]
        self._drawing = [False] * n_hands

    def clear(self):
        for d in self._trail:
            d.clear()
        self.clear_strokes()

    def clear_strokes(self):
        for s in self._strokes:
            s.clear()
        self._drawing = [False] * len(self._drawing)

    # ------------------------------------------------------------------------
    def update(self, hands, t, dt):
        for i, hand in enumerate(hands):
            trail = self._trail[i]

            if hand.visible and (hand.gesture in (POINT, PEACE, PINCH)
                                 or hand.speed > 1.1):
                p = hand.index_tip.copy()
                if not trail or np.linalg.norm(p - trail[-1][0]) > 1.5:
                    trail.append((p, 0.0))

            # Age the samples and retire the expired ones.
            self._trail[i] = deque(
                ((p, age + dt) for p, age in trail if age + dt < TRAIL_TTL),
                maxlen=config.TRAIL_LENGTH,
            )

            # Draw mode: two fingers up lays down a persisting stroke.
            drawing = hand.visible and hand.gesture == PEACE
            strokes = self._strokes[i]
            if drawing:
                # A stroke can also vanish underneath us: the ageing pass below
                # discards any that has decayed to fewer than two points.
                if not self._drawing[i] or not strokes:
                    strokes.append([])
                    if len(strokes) > 14:
                        strokes.pop(0)
                p = hand.index_tip.copy()
                cur = strokes[-1]
                if not cur or np.linalg.norm(p - cur[-1][0]) > 3.0:
                    cur.append((p, 0.0))
            self._drawing[i] = drawing

            for si in range(len(strokes) - 1, -1, -1):
                aged = [(p, age + dt) for p, age in strokes[si]
                        if age + dt < STROKE_TTL]
                if len(aged) < 2:
                    strokes.pop(si)
                else:
                    strokes[si] = aged

    def draw(self, ov, hands, t):
        for i, hand in enumerate(hands):
            trail = self._trail[i]
            if len(trail) >= 2:
                pts = [p for p, _ in trail]
                ages = [a for _, a in trail]
                gain = clamp(0.22 + hand.speed * 0.30) * max(hand.presence, 0.0)
                width = 1 + int(clamp(hand.speed / 2.4) * 2)
                _ribbon(ov, pts, ages, TRAIL_TTL, config.WHITE, gain, width)
                ov.dot(pts[-1], 1.6, config.WHITE, 0.5 * gain)

            for stroke in self._strokes[i]:
                pts = [p for p, _ in stroke]
                ages = [a for _, a in stroke]
                _ribbon(ov, pts, ages, STROKE_TTL, config.CYAN, 0.62, 2)
