"""The cold-start title sequence.

Four overlapping stages over roughly four seconds, all driven off one clock so
they cross-fade rather than cut:

  sweep    a scan line crosses the black, leaving horizontal raster bars
  title    UDULAIW resolves letter by letter out of a glitching character
           cipher, each letter landing with a chromatic split and a shock ring
  log      boot lines type themselves down the left margin
  release  the frame is handed over -- brackets snap to the corners, the veil
           lifts and the camera bleeds through

`veil` is the one value the app needs: 1.0 while the sequence owns the screen,
falling to 0.0 as the camera comes up. Everything else is drawn straight onto
the normal overlay, so the title gets the same bloom as the rest of the
interface for free.

Any key skips it.
"""
import random

import config
from utils.smoothing import clamp, smoothstep

# Characters a letter cycles through before it settles. Deliberately dense and
# angular -- rounded glyphs read as a typo rather than as noise.
_CIPHER = "#%&$@8XKZW4NVM70/\\|+=<>*"

_TITLE_SCALE = 2.85
_TITLE_TRACK = 17.0          # extra pixels between letters
_TITLE_WEIGHT = 3            # stroke thickness of the title glyphs


class BootSequence:
    def __init__(self, width, height, duration=None):
        self.w = width
        self.h = height
        self.duration = float(duration or config.BOOT_DURATION)
        self.t = 0.0
        self.done = False
        self._rng = random.Random(4)
        self._glitch = {}
        self._glitch_at = -1.0
        self._layout = None
        self._skipped = False

    # ------------------------------------------------------------------------
    @property
    def veil(self):
        """How much of the camera to hold back, 1 = fully black."""
        if self.done:
            return 0.0
        k = self.t / self.duration
        # Solid until the last quarter, then lifted with an ease so the image
        # arrives rather than appearing.
        return 1.0 - smoothstep(0.74, 1.0, k)

    @property
    def running(self):
        return not self.done

    def skip(self):
        if not self.done:
            self._skipped = True
            # Jump into the release stage rather than cutting outright; a hard
            # cut to a live camera is jarring after a black screen.
            self.t = max(self.t, self.duration * 0.80)

    def update(self, dt):
        self.t += dt
        if self.t >= self.duration:
            self.done = True
        return self.done

    # ------------------------------------------------------------------------
    def draw(self, ov, t):
        if self.done:
            return
        k = clamp(self.t / self.duration)
        self._sweep(ov, k)
        self._title(ov, k, t)
        if not self._skipped:
            self._log(ov, k)
        self._release(ov, k, t)

    # ---------------------------------------------------------------- stages --
    def _sweep(self, ov, k):
        """Raster bars, plus one bright line crossing early on."""
        fade = 1.0 - smoothstep(0.62, 1.0, k)
        if fade <= 0.01:
            return
        step = 4
        a = 0.030 * fade
        for y in range(0, self.h, step):
            ov.line((0, y), (self.w, y), config.WHITE, a, 1)

        s = smoothstep(0.0, 0.30, k)
        if s < 1.0:
            y = self.h * (1.0 - (1.0 - s) ** 2)
            ov.line((0, y), (self.w, y), config.CYAN, 0.55 * fade, 2)
            ov.line((0, y + 6), (self.w, y + 6), config.CYAN, 0.16 * fade, 1)
            label = "INITIALISING"
            ov.text((28, min(self.h - 12, y + 22)), label, config.DIM,
                    0.7 * fade, 0.40, 1)

    # ------------------------------------------------------------------------
    def _title_layout(self, ov):
        """Per-letter x offsets, measured once and cached."""
        if self._layout is not None:
            return self._layout
        text = config.BOOT_TEXT
        widths = [ov.measure(c, _TITLE_SCALE, _TITLE_WEIGHT, mono=False)[0] for c in text]
        total = sum(widths) + _TITLE_TRACK * (len(text) - 1)
        x = (self.w - total) * 0.5
        xs = []
        for wdt in widths:
            xs.append(x)
            x += wdt + _TITLE_TRACK
        _, height = ov.measure(text, _TITLE_SCALE, _TITLE_WEIGHT, mono=False)
        self._layout = (text, xs, widths, height, total)
        return self._layout

    def _title(self, ov, k, t):
        text, xs, widths, height, total = self._title_layout(ov)
        n = len(text)
        y = self.h * 0.48 + height * 0.5

        # Letters resolve left to right between these two points on the clock.
        lo, hi = 0.08, 0.56
        # Refresh the cipher a few times a second, not every frame: glyphs that
        # change at 60 Hz just read as a grey smear.
        if self.t - self._glitch_at > 0.055:
            self._glitch_at = self.t
            self._glitch = {i: self._rng.choice(_CIPHER) for i in range(n)}

        for i, ch in enumerate(text):
            start = lo + (hi - lo) * (i / max(n, 1))
            settle = start + 0.16
            p = clamp((k - start) / max(settle - start, 1e-3))
            if p <= 0.0:
                continue
            x = xs[i]
            if p < 1.0:
                glyph = self._glitch.get(i, ch)
                jitter = (1.0 - p) * 7.0
                gx = x + self._rng.uniform(-jitter, jitter)
                gy = y + self._rng.uniform(-jitter, jitter)
                ov.text((gx, gy), glyph, config.DIM, 0.35 + 0.5 * p,
                        _TITLE_SCALE, _TITLE_WEIGHT, mono=False)
                continue

            # Landed. Chromatic split decays over the first moments, then the
            # letter just breathes.
            age = k - settle
            split = max(0.0, 1.0 - age * 6.0)
            hold = 1.0 - smoothstep(0.88, 1.0, k) * 0.15
            if split > 0.01:
                off = 1.0 + split * 5.0
                ov.text((x - off, y), ch, config.CYAN, 0.55 * split,
                        _TITLE_SCALE, _TITLE_WEIGHT, mono=False)
                ov.text((x + off, y), ch, config.AMBER, 0.45 * split,
                        _TITLE_SCALE, _TITLE_WEIGHT, mono=False)
            ov.text((x, y), ch, config.WHITE, hold, _TITLE_SCALE, _TITLE_WEIGHT, mono=False)

            # One shock ring per letter as it lands.
            if age < 0.42:
                r = 10.0 + age * 210.0
                cx = x + widths[i] * 0.5
                cy = y - height * 0.42
                ov.circle((cx, cy), r, config.CYAN,
                          (1.0 - age / 0.42) ** 2 * 0.45, 1)

        # Underline draws itself in beneath the finished word.
        u = smoothstep(0.40, 0.72, k)
        if u > 0.0:
            x0 = xs[0]
            ov.line((x0, y + 20), (x0 + total * u, y + 20), config.CYAN,
                    0.7, 1)

        # Subtitle, tracked out letter by letter.
        s = smoothstep(0.56, 0.78, k)
        if s > 0.0:
            sub = config.BOOT_SUB
            cut = max(1, int(len(sub) * s))
            shown = sub[:cut]
            sw, _ = ov.measure(shown, 0.52)
            ov.text(((self.w - sw) * 0.5, y + 52), shown, config.DIM,
                    0.85, 0.52, 1)

    # ------------------------------------------------------------------------
    def _log(self, ov, k):
        lines = config.BOOT_LOG
        # Faded out before the handover: the status line lives in this corner
        # and the two would otherwise print over each other.
        gain = 1.0 - smoothstep(0.78, 0.94, k)
        if gain <= 0.01:
            return
        s = smoothstep(0.22, 0.80, k)
        shown = clamp(s) * len(lines)
        y = self.h - 96 - (len(lines) - 1) * 16
        for i, line in enumerate(lines):
            p = shown - i
            if p <= 0.0:
                continue
            cut = max(1, int(len(line) * min(1.0, p * 2.2)))
            partial = line[:cut]
            done = cut >= len(line)
            a = (0.34 if done else 0.80) * gain
            ov.text((28, y + i * 16), partial, config.DIM, a, 0.34, 1)
            if done:
                ov.text((300, y + i * 16), "OK", config.CYAN, 0.55 * gain,
                        0.34, 1)
            else:
                ov.dot((28 + ov.measure(partial, 0.34)[0] + 4,
                        y + i * 16 - 3), 2.0, config.AMBER, 0.9 * gain)

    # ------------------------------------------------------------------------
    def _release(self, ov, k, t):
        """Corner brackets snap in, then the handover flash."""
        b = smoothstep(0.62, 0.86, k)
        if b <= 0.0:
            return
        m = 22 + (1.0 - b) * 90.0
        arm = 16 + 40.0 * (1.0 - b)
        w, h = self.w, self.h
        for (cx, cy), sx, sy in (((m, m), 1, 1), ((w - m, m), -1, 1),
                                 ((w - m, h - m), -1, -1), ((m, h - m), 1, -1)):
            ov.line((cx, cy), (cx + sx * arm, cy), config.WHITE, 0.55 * b, 1)
            ov.line((cx, cy), (cx, cy + sy * arm), config.WHITE, 0.55 * b, 1)

        f = smoothstep(0.80, 0.90, k) * (1.0 - smoothstep(0.90, 1.0, k))
        if f > 0.01:
            msg = "SYSTEM ONLINE"
            mw, _ = ov.measure(msg, 0.60)
            ov.text(((self.w - mw) * 0.5, self.h * 0.66), msg, config.CYAN,
                    f, 0.60, 1, shift=True)
            r = 60.0 + (1.0 - f) * 500.0
            ov.circle((self.w * 0.5, self.h * 0.48), r, config.WHITE,
                      f * 0.18, 1)

        # A last wash of raster as the camera bleeds through.
        g = smoothstep(0.86, 1.0, k)
        if 0.0 < g < 1.0:
            span = int(self.h * g)
            for _ in range(6):
                y = self._rng.randrange(0, max(span, 1))
                ov.line((0, y), (self.w, y), config.CYAN,
                        0.10 * (1.0 - g), 1)
