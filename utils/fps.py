"""Frame timing."""
import time
from collections import deque


class FrameClock:
    def __init__(self, window=45):
        self._t = time.perf_counter()
        self._samples = deque(maxlen=window)
        self.dt = 1.0 / 60.0
        self.elapsed = 0.0

    def tick(self):
        now = time.perf_counter()
        self.dt = min(now - self._t, 0.1)   # clamp so a stall can't explode physics
        self._t = now
        self.elapsed += self.dt
        self._samples.append(self.dt)
        return self.dt

    @property
    def fps(self):
        if not self._samples:
            return 0.0
        mean = sum(self._samples) / len(self._samples)
        return 1.0 / mean if mean > 0 else 0.0
