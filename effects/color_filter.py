"""The two-hand pinch bar: a colour filter you draw in the air.

Pinch thumb-to-index on BOTH hands at once and a line snaps between the two
pinch points. That line is the control surface:

    length   scrubs the filter list  (hands together = OFF, wide apart = last)
    angle    dials strength          (level = 50%, tilted = 0% .. 100%)

Release either pinch and the pick latches. The filter itself is applied by
CameraGrade — this module only decides *what* is selected and draws the
instrument, so the grade never has to know a gesture exists.

Two details carry the feel:

  * The scrub is smoothed and the pick snaps with hysteresis. Raw hand distance
    crosses a tick boundary several times a second while you hold still, and
    without the deadband the filter strobes between neighbours.
  * The whole bar is drawn from the *smoothed* endpoints, not the raw pinch
    points, so the line reads as a rigid object held between the hands rather
    than two dots connected by a wire.
"""
import math

import numpy as np

import config
from utils.smoothing import Ema, clamp, smoothstep

# Blank space kept clear at each end so the bar does not grow out of the
# fingers themselves.
_END_INSET = 26.0


def _lerp_pt(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


class FilterBar:
    """Selection state for the colour filter, plus its on-screen instrument."""

    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.index = 0
        self.strength = 0.0
        self.armed = False
        # Continuous position along the list, 0 .. n-1. Kept separate from
        # `index` so the smoothing has somewhere to live between snaps.
        self._scrub = 0.0
        self._scrub_ema = Ema(config.FILTER_SMOOTH, 0.0)
        self._tilt_ema = Ema(config.FILTER_SMOOTH, 0.5)
        self._vis = Ema(0.10, 0.0)
        self._latched = Ema(0.16, 0.0)
        self._a = None
        self._b = None
        self._flash = 0.0
        self.events = []

    def resize(self, width, height):
        self.w, self.h = width, height

    def reset(self):
        self.index = 0
        self.strength = 0.0
        self.armed = False
        self._scrub = 0.0
        self._scrub_ema.value = 0.0
        self._tilt_ema.value = 0.5
        self._a = self._b = None
        self._flash = 0.0

    # ------------------------------------------------------------------------
    def cycle(self, step=1):
        """Keyboard fallback for picking a filter without the gesture."""
        self.index = (self.index + step) % len(config.FILTERS)
        if self.index != 0 and self.strength <= 0.02:
            self.strength = 0.75
        self._scrub = float(self.index)
        self._scrub_ema.value = self._scrub
        self._flash = 1.0
        return self.label

    def select(self, name, strength=0.8):
        """Latch a filter by name. Unknown names raise, so a typo on the
        command line fails loudly instead of silently starting unfiltered."""
        key = str(name).strip().upper()
        for i, entry in enumerate(config.FILTERS):
            if entry[0] == key:
                self.index = i
                self.strength = 0.0 if i == 0 else float(strength)
                self._scrub = float(i)
                self._scrub_ema.value = self._scrub
                return self.label
        known = ", ".join(f[0] for f in config.FILTERS)
        raise ValueError(f"unknown filter {name!r}; expected one of: {known}")

    def clear(self):
        self.index = 0
        self.strength = 0.0
        self._scrub = 0.0
        self._scrub_ema.value = 0.0
        self._flash = 1.0

    @property
    def label(self):
        return config.FILTERS[self.index][0]

    @property
    def swatch(self):
        return config.FILTERS[self.index][3]

    @property
    def active(self):
        return self.index != 0 and self.strength > 0.02

    @property
    def anchors(self):
        """The two bar endpoints, or an empty tuple before the first arm."""
        if self._a is None or self._b is None:
            return ()
        return (self._a, self._b)

    # ------------------------------------------------------------------ update
    def update(self, hands, t, dt, enabled=True):
        """Returns queued events for the app to toast.

        `enabled=False` yields the two-hand pinch to another instrument -- the
        ORBIX viewer claims it for scaling while it is open. The bar disarms
        without latching anything, so whatever filter was already selected
        stays exactly as it was and comes straight back when control returns.
        """
        self.events.clear()
        self._flash = max(0.0, self._flash - dt * 1.6)

        if not enabled:
            self.armed = False
            self._vis(0.0, dt)
            self._latched(1.0 if self.active else 0.0, dt)
            return self.events

        pinched = [h for h in hands if h.visible and h.pinch]
        engaged = len(pinched) >= 2

        if engaged:
            a = np.asarray(pinched[0].pinch_point, np.float32)
            b = np.asarray(pinched[1].pinch_point, np.float32)
            # Keep left-to-right stable so the tilt sign does not flip when the
            # hands cross or the tracker swaps their slots.
            if a[0] > b[0]:
                a, b = b, a
            self._a, self._b = a, b

            d = b - a
            gap = float(np.hypot(d[0], d[1])) / max(self.w, 1)
            n = len(config.FILTERS)
            k = smoothstep(config.FILTER_GAP_MIN, config.FILTER_GAP_MAX, gap)
            if not self.armed:
                # Snap on the arming frame. Gliding in from the last session's
                # value would sweep the preview through every filter in between.
                self._scrub_ema.value = k * (n - 1)
                self._tilt_ema.value = None
            self._scrub = float(self._scrub_ema(k * (n - 1), dt))

            # Tilt: the bar's angle off horizontal, clamped to the usable range.
            # Screen y grows downward, so negate to make "raise the right hand"
            # read as more strength.
            ang = math.degrees(math.atan2(-float(d[1]), float(d[0])))
            tilt = clamp(0.5 + ang / (2.0 * config.FILTER_TILT_RANGE))
            self.strength = float(self._tilt_ema(tilt, dt))

            # Hysteresis: only move the pick once the scrub is clearly past the
            # neighbouring tick.
            want = int(round(self._scrub))
            if abs(self._scrub - self.index) > config.FILTER_SNAP_MARGIN:
                if want != self.index:
                    self.index = max(0, min(n - 1, want))
                    self._flash = 1.0

            if not self.armed:
                self.armed = True
                self.events.append(("filter_arm", None))
        elif self.armed:
            self.armed = False
            self._flash = 1.0
            if self.index == 0 or self.strength <= 0.02:
                self.index = 0
                self.strength = 0.0
                self.events.append(("filter_set", "FILTER OFF"))
            else:
                self.events.append(
                    ("filter_set",
                     f"{self.label}  {self.strength * 100:.0f}%"))

        self._vis(1.0 if engaged else 0.0, dt)
        self._latched(1.0 if self.active else 0.0, dt)
        return self.events

    # -------------------------------------------------------------------- draw
    def draw(self, ov, t):
        vis = float(self._vis.value or 0.0)
        if vis > 0.01 and self._a is not None:
            self._draw_bar(ov, vis, t)
        chip = float(self._latched.value or 0.0)
        if chip > 0.01:
            self._draw_chip(ov, chip)

    def _draw_bar(self, ov, vis, t):
        a, b = self._a, self._b
        d = b - a
        length = float(np.hypot(d[0], d[1]))
        if length < 2.0 * _END_INSET:
            # Hands too close to hold a bar: show the pick as a pair of rings
            # so the gesture still reads as engaged.
            for p in (a, b):
                ov.circle(p, 12, config.DIM, 0.5 * vis, 1)
            return

        u = d / length
        perp = np.array([-u[1], u[0]], np.float32)
        p0 = a + u * _END_INSET
        p1 = b - u * _END_INSET
        n = len(config.FILTERS)

        # Segments, one per filter, each in its own swatch colour.
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            s0 = _lerp_pt(p0, p1, t0 + 0.012)
            s1 = _lerp_pt(p0, p1, t1 - 0.012)
            hot = (i == self.index)
            color = config.FILTERS[i][3]
            ov.line(s0, s1, color, (0.95 if hot else 0.38) * vis,
                    4 if hot else 2)

        # Ticks at every filter position, taller for the active one.
        for i in range(n):
            c = np.asarray(_lerp_pt(p0, p1, (i + 0.5) / n), np.float32)
            hot = (i == self.index)
            hgt = 11.0 if hot else 5.0
            ov.line(c - perp * hgt, c + perp * hgt, config.FILTERS[i][3],
                    (0.9 if hot else 0.22) * vis, 2 if hot else 1)

        # Endpoint rings sit on the pinch points themselves.
        pulse = 1.0 + 0.10 * math.sin(t * 5.0)
        for p in (a, b):
            ov.circle(p, 13 * pulse, config.WHITE, 0.75 * vis, 1)
            ov.dot(p, 2.2, self.swatch, 0.95 * vis)

        self._draw_readout(ov, p0, p1, perp, vis)

    def _draw_readout(self, ov, p0, p1, perp, vis):
        mid = np.asarray(_lerp_pt(p0, p1, 0.5), np.float32)
        # Push the caption to whichever side of the bar has room, so a bar held
        # near the top of the frame does not write its label off-screen.
        side = -1.0 if mid[1] > self.h * 0.30 else 1.0
        anchor = mid + perp * (34.0 * side)

        name = self.label
        pct = f"{self.strength * 100:.0f}%"
        tw, _ = ov.measure(name, 0.50)
        flash = 0.28 * self._flash
        ov.text((anchor[0] - tw * 0.5, anchor[1]), name, self.swatch,
                min(1.0, (0.92 + flash)) * vis, 0.50, 1, shift=True)

        if self.index != 0:
            pw, _ = ov.measure(pct, 0.40)
            ov.text((anchor[0] - pw * 0.5, anchor[1] + 17), pct, config.WHITE,
                    0.7 * vis, 0.40, 1)
            # Strength read as a short bar under the percentage.
            bw = 46.0
            bx, by = anchor[0] - bw * 0.5, anchor[1] + 25
            ov.line((bx, by), (bx + bw, by), config.DIM, 0.30 * vis, 1)
            ov.line((bx, by), (bx + bw * self.strength, by), self.swatch,
                    0.85 * vis, 2)

        hint = f"{self.index + 1}/{len(config.FILTERS)}"
        hw, _ = ov.measure(hint, 0.36)
        ov.text((anchor[0] - hw * 0.5, anchor[1] - 16), hint, config.DIM,
                0.55 * vis, 0.36, 1)

    def _draw_chip(self, ov, alpha):
        """Persistent bottom-right marker for whatever filter is latched."""
        text = f"FLT {self.label} {self.strength * 100:.0f}%"
        tw, _ = ov.measure(text, 0.40)
        x, y = self.w - tw - 28, self.h - 26
        ov.dot((x - 12, y - 4), 3.0, self.swatch, 0.9 * alpha)
        ov.text((x, y), text, config.WHITE, 0.8 * alpha, 0.40, 1)
