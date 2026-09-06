"""Signal smoothing.

MediaPipe landmarks are noisy at rest and laggy under naive low-pass. The One
Euro filter adapts its cutoff to speed: heavy smoothing when the hand is still,
light smoothing when it moves, which is what makes the overlay feel welded to
the hand instead of swimming behind it.
"""
import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Vectorised One Euro filter — operates on a whole landmark array."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = None
        self._t = None

    def reset(self):
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float32)
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x

        dt = t - self._t
        if dt <= 1e-5:
            dt = 1.0 / 60.0
        self._t = t

        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx += a_d * (dx - self._dx)

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        # per-element alpha
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self._x += a * (x - self._x)
        return self._x


class Ema:
    """Scalar / small-vector exponential average with a half-life in seconds."""

    def __init__(self, half_life=0.12, initial=None):
        self.k = math.log(2.0) / max(half_life, 1e-4)
        self.value = initial

    def __call__(self, target, dt):
        if self.value is None:
            self.value = target
            return self.value
        a = 1.0 - math.exp(-self.k * max(dt, 1e-4))
        self.value += (target - self.value) * a
        return self.value


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else (hi if v > hi else v)


def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / max(edge1 - edge0, 1e-6))
    return t * t * (3.0 - 2.0 * t)
