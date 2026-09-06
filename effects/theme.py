"""Runtime accent palette.

The interface has exactly two accent slots -- `config.CYAN` for structure and
`config.AMBER` for anything hot or active -- and every effect reads them
through the config module at draw time rather than binding them at import. That
one property is what makes the palette swappable while the app runs: this class
writes new values into those two names and the entire overlay restyles on the
next frame, with no effect needing to know a theme system exists.

The change is cross-faded rather than switched. A hard cut through eight
overlays at once reads as a glitch; a 0.2 s fade reads as the interface
retuning itself.

This is deliberately a different axis from the colour filter in
`effects/color_filter.py`. That one grades the *camera*; this one recolours the
*instrumentation*. They compose, and neither replaces the other.
"""
import numpy as np

import config
from utils.smoothing import Ema


def _tuple(arr):
    return (float(arr[0]), float(arr[1]), float(arr[2]))


class ThemePalette:
    def __init__(self, index=0):
        self.index = int(index) % len(config.THEMES)
        accent, hot = config.THEMES[self.index][1:3]
        self._accent = np.asarray(accent, np.float32)
        self._hot = np.asarray(hot, np.float32)
        self._blend = Ema(config.THEME_FADE, 0.0)
        self._from = (self._accent.copy(), self._hot.copy())
        self.flash = 0.0
        self.apply()

    # ------------------------------------------------------------------------
    @property
    def name(self):
        return config.THEMES[self.index][0]

    @property
    def accent(self):
        """The live, mid-fade accent colour as a BGR tuple."""
        return _tuple(self._accent)

    @property
    def hot(self):
        return _tuple(self._hot)

    @property
    def count(self):
        return len(config.THEMES)

    @staticmethod
    def swatch(i):
        """Target accent of theme `i`, for drawing the picker."""
        return config.THEMES[i % len(config.THEMES)][1]

    @staticmethod
    def hot_swatch(i):
        return config.THEMES[i % len(config.THEMES)][2]

    # ------------------------------------------------------------------------
    def select(self, index, instant=False):
        index = int(index) % len(config.THEMES)
        if index == self.index and not instant:
            return self.name
        self.index = index
        self._from = (self._accent.copy(), self._hot.copy())
        self._blend.value = 1.0 if instant else 0.0
        self.flash = 1.0
        if instant:
            self.update(0.0)
        return self.name

    def cycle(self, step=1):
        return self.select(self.index + step)

    # ------------------------------------------------------------------------
    def update(self, dt):
        k = float(self._blend(1.0, dt))
        target_a = np.asarray(config.THEMES[self.index][1], np.float32)
        target_h = np.asarray(config.THEMES[self.index][2], np.float32)
        self._accent = self._from[0] + (target_a - self._from[0]) * k
        self._hot = self._from[1] + (target_h - self._from[1]) * k
        self.flash = max(0.0, self.flash - dt * 1.7)
        self.apply()

    def apply(self):
        """Publish the current colours into config, where the effects read."""
        config.CYAN = _tuple(self._accent)
        config.AMBER = _tuple(self._hot)
