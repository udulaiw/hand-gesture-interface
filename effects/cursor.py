"""Index-fingertip cursor and the pinch interaction.

The cursor is a reticle, not a pointer graphic: a ring, four ticks, a centre
dot, and a slowly counter-rotating pair of arcs so it reads as instrumentation.
Pinching draws a ring between the finger pads whose radius tracks the real gap,
with a progress arc that closes as the pinch tightens and a one-shot
confirmation flash on the transition.
"""
import math

import numpy as np

import config
from tracking.gestures import PEACE, POINT
from utils.smoothing import Ema, smoothstep


class _Flash:
    __slots__ = ("t", "pos", "life", "color")

    def __init__(self, pos, life=0.55, color=config.AMBER):
        self.pos = np.asarray(pos, np.float32)
        self.t = 0.0
        self.life = life
        self.color = color


class Cursor:
    def __init__(self, n_hands):
        self._vis = [Ema(0.11, 0.0) for _ in range(n_hands)]
        self._pinch = [Ema(0.07, 0.0) for _ in range(n_hands)]
        self._flashes = []

    def reset(self):
        self._flashes.clear()
        for e in self._vis:
            e.value = 0.0
        for e in self._pinch:
            e.value = 0.0

    def confirm(self, pos, color=config.AMBER):
        self._flashes.append(_Flash(pos, color=color))
        if len(self._flashes) > 8:
            self._flashes.pop(0)

    # ------------------------------------------------------------------------
    def update(self, dt):
        for f in self._flashes:
            f.t += dt
        self._flashes = [f for f in self._flashes if f.t < f.life]

    def draw(self, ov, hands, t, dt):
        for i, hand in enumerate(hands):
            want = 0.0
            if hand.visible:
                if hand.gesture in (POINT, PEACE):
                    want = 1.0
                elif hand.extension[1] > 0.75:
                    want = 0.34
            a = self._vis[i](want, dt) * hand.presence
            if a > 0.02:
                self._reticle(ov, hand.index_tip, a, t, hand.size)

            pa = self._pinch[i](1.0 if hand.pinch else 0.0, dt) * hand.presence
            if pa > 0.02:
                self._pinch_ring(ov, hand, pa, t)

        for f in self._flashes:
            k = f.t / f.life
            r = 8.0 + k * 34.0
            ov.circle(f.pos, r, f.color, (1.0 - k) ** 2 * 0.8, 1)
            ov.circle(f.pos, r * 0.55, config.WHITE, (1.0 - k) ** 3 * 0.5, 1)

    # ------------------------------------------------------------------------
    @staticmethod
    def _reticle(ov, p, a, t, size):
        r = max(6.0, size * 0.11)
        ov.circle(p, r, config.WHITE, 0.55 * a, 1)
        ov.dot(p, 1.5, config.AMBER, 0.9 * a)
        tick = r * 0.42
        for k in range(4):
            ang = k * math.pi / 2.0
            d = np.array([math.cos(ang), math.sin(ang)], np.float32)
            ov.line(p + d * (r + 2.5), p + d * (r + 2.5 + tick),
                    config.WHITE, 0.42 * a, 1)
        spin = -t * 46.0
        ov.arc(p, r * 1.72, spin, spin + 72, config.CYAN, 0.5 * a, 1)
        ov.arc(p, r * 1.72, spin + 180, spin + 252, config.CYAN, 0.5 * a, 1)

    @staticmethod
    def _pinch_ring(ov, hand, a, t):
        p = hand.pinch_point
        gap = float(np.linalg.norm(hand.thumb_tip - hand.index_tip))
        # Sits between the finger pads, but never grows into a HUD element.
        r = min(max(5.0, gap * 0.5 + 4.0), hand.size * 0.30)
        closed = smoothstep(config.PINCH_OFF, config.PINCH_ON, hand.pinch_dist)

        ov.line(hand.thumb_tip, hand.index_tip, config.WHITE, 0.30 * a, 1)
        ov.circle(p, r, config.AMBER, (0.45 + 0.45 * closed) * a, 1)
        sweep = 360.0 * closed
        ov.arc(p, r + 4.0, -90.0, -90.0 + sweep, config.AMBER, 0.85 * a, 2)
        ov.dot(p, 1.8 + 1.6 * closed, config.WHITE, (0.5 + 0.5 * closed) * a)

        pulse = 0.5 + 0.5 * math.sin(t * 6.0)
        ov.circle(p, r + 10.0 + 5.0 * pulse, config.AMBER,
                  0.16 * a * closed * (1.0 - pulse), 1)
