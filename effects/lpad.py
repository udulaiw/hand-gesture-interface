"""The L-frame accent pad.

Hold one hand in an L -- thumb out, index up, the other three folded -- and a
quarter-ring of swatches unfolds into the empty wedge between the thumb and the
index finger. It is drawn in that wedge and nowhere else: the arc starts on the
thumb's axis, ends on the index's, and rides the corner where the two meet, so
the panel is anchored to the hand's own geometry and turns with it.

There are two ways to drive it, and both are live at once:

  dial     twist the wrist. The roll angle accumulated since the pad opened
           scrubs the highlight around the arc, and closing the L into a pinch
           commits the pick. This is the one-handed path.
  point    put the *other* hand's index fingertip on a swatch. Direct hits win
           over the dial immediately, and a short dwell commits without needing
           a second gesture.

Commit writes the accent through to `ThemePalette`, which is what actually
restyles the interface. Releasing the pose without committing restores the
theme that was live when the pad opened, so browsing is free.

This pad and the two-hand colour bar are separate instruments on purpose: the
bar grades the camera image, the pad recolours the overlay. Neither one
replaces or disables the other.
"""
import math

import numpy as np

import config
from tracking.gestures import L_SHAPE
from utils.smoothing import Ema, clamp, smoothstep


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class LPad:
    """Selection state and rendering for the L-frame accent picker."""

    def __init__(self, palette):
        self.palette = palette
        self.open_amount = Ema(config.LPAD_OPEN, 0.0)
        self.hand_index = None
        self.highlight = palette.index
        self.events = []

        self._entry_theme = palette.index
        self._roll_ref = None       # wrist roll captured when the pad opened
        self._roll_accum = 0.0
        self._prev_roll = None
        self._dwell = 0.0
        self._dwell_on = None
        self._pointer = None        # screen position of the other hand's tip
        self._committed = 0.0       # flash timer after a successful pick
        self._geom = None           # (corner, a0, a1, r0, r1) from last frame

    # ------------------------------------------------------------------------
    def reset(self):
        self.open_amount.value = 0.0
        self.hand_index = None
        self._roll_ref = None
        self._prev_roll = None
        self._roll_accum = 0.0
        self._dwell = 0.0
        self._dwell_on = None
        self._pointer = None
        self._geom = None

    @property
    def active(self):
        return self.hand_index is not None

    @property
    def visible(self):
        return float(self.open_amount.value or 0.0) > 0.01

    # ------------------------------------------------------------------ update
    def update(self, hands, t, dt):
        """Returns queued ('toast', text) events for the app."""
        self.events.clear()
        self._committed = max(0.0, self._committed - dt * 1.8)

        holder = self._find_holder(hands)
        if holder is None:
            if self.hand_index is not None:
                self._close(hands, committed=False)
            self.open_amount(0.0, dt)
            return self.events

        i, hand = holder
        if self.hand_index != i:
            self._open(i, hand)

        self._track_dial(hand, dt)
        self._track_pointer(hands, i, hand, dt)
        self.palette.select(self.highlight)     # live preview while browsing
        self.open_amount(1.0, dt)
        return self.events

    def _find_holder(self, hands):
        """The hand currently making the L, preferring the one already in use."""
        candidates = [(i, h) for i, h in enumerate(hands)
                      if h.visible and h.gesture == L_SHAPE]
        if not candidates:
            return None
        for i, h in candidates:
            if i == self.hand_index:
                return i, h
        return candidates[0]

    def _open(self, i, hand):
        self.hand_index = i
        self._entry_theme = self.palette.index
        self.highlight = self.palette.index
        self._roll_ref = float(hand.rotation)
        self._prev_roll = float(hand.rotation)
        self._roll_accum = 0.0
        self._dwell = 0.0
        self._dwell_on = None

    def _close(self, hands, committed):
        if not committed:
            # Browsing was free: put back whatever was live before the pad
            # opened. A commit has already made the pick permanent.
            self.palette.select(self._entry_theme)
        self.hand_index = None
        self._roll_ref = None
        self._prev_roll = None
        self._pointer = None
        self._dwell = 0.0
        self._dwell_on = None

    # ------------------------------------------------------------------- dial
    def _track_dial(self, hand, dt):
        """Wrist roll since the pad opened, converted to a swatch index.

        Accumulated from per-frame deltas rather than read as an absolute angle,
        so the dial can be turned past +-180 degrees without wrapping around and
        jumping to the far end of the ring.
        """
        roll = float(hand.rotation)
        if self._prev_roll is not None:
            self._roll_accum += _wrap_pi(roll - self._prev_roll)
        self._prev_roll = roll

        if self._pointer is not None:
            return      # the pointing hand is driving; leave the dial parked

        step = math.radians(max(config.LPAD_DIAL_STEP, 1.0))
        n = self.palette.count
        moved = int(round(self._roll_accum / step))
        self.highlight = (self._entry_theme + moved) % n

    # ---------------------------------------------------------------- pointer
    def _track_pointer(self, hands, holder_index, hand, dt):
        """Other-hand fingertip hit test against the swatch arc."""
        self._pointer = None
        geom = self._geom
        if geom is None:
            return
        corner, a0, span, r0, r1 = geom

        best = None
        for j, other in enumerate(hands):
            if j == holder_index or not other.visible:
                continue
            if float(other.extension[1]) < 0.55:
                continue                     # index must actually be pointing
            p = np.asarray(other.index_tip, np.float32)
            d = p - corner
            rad = float(np.hypot(d[0], d[1]))
            if not (r0 * 0.80 <= rad <= r1 * 1.20):
                continue
            # Angle of the fingertip measured inside the wedge, 0..1.
            k = _wrap_pi(math.atan2(float(d[1]), float(d[0])) - a0) / span
            if not (-0.06 <= k <= 1.06):
                continue
            best = (j, other, p, clamp(k))
            break

        if best is None:
            self._dwell = 0.0
            self._dwell_on = None
            return

        j, other, p, k = best
        self._pointer = p
        n = self.palette.count
        idx = min(n - 1, int(k * n))
        self.highlight = idx

        # Dwell, or an immediate pinch from the pointing hand, both commit.
        if self._dwell_on != idx:
            self._dwell_on = idx
            self._dwell = 0.0
        self._dwell += dt
        if other.pinch or self._dwell >= config.LPAD_DWELL:
            self.commit()

    # ----------------------------------------------------------------- commit
    def commit(self):
        """Latch the highlighted accent. Called by dwell, or by a pinch."""
        name = self.palette.select(self.highlight, instant=False)
        self._entry_theme = self.highlight
        self._committed = 1.0
        self._dwell = 0.0
        self._dwell_on = None
        self.events.append(("toast", "ACCENT  " + name))
        return name

    def note_pinch(self, hand_index):
        """The app routes an index pinch here so closing the L commits.

        The L pose ends the instant the finger and thumb meet, so by the time a
        pinch is detected the pad is already on its way out. Committing from the
        event, rather than from the pose, is what makes 'twist then pinch' work.
        """
        if self.hand_index == hand_index and self.visible:
            self.commit()
            return True
        return False

    # -------------------------------------------------------------------- draw
    def draw(self, ov, hands, t, dt):
        a = float(self.open_amount.value or 0.0)
        if a < 0.012:
            self._geom = None
            return
        hand = None
        if self.hand_index is not None and self.hand_index < len(hands):
            h = hands[self.hand_index]
            if h.visible:
                hand = h
        if hand is None:
            # Fading out after the hand went away: keep the last geometry so
            # the pad collapses in place instead of snapping to the origin.
            if self._geom is None:
                return
            self._draw_ring(ov, a * 0.6, t, None)
            return

        corner, u_thumb, u_index = hand.l_basis()
        a_thumb = math.atan2(float(u_thumb[1]), float(u_thumb[0]))
        a_index = math.atan2(float(u_index[1]), float(u_index[0]))
        wedge = _wrap_pi(a_index - a_thumb)
        if abs(wedge) < 1e-3:
            return

        # Widen the sweep about the wedge's bisector, keeping its direction so
        # the fan always unrolls from the thumb toward the index finger.
        sign = 1.0 if wedge >= 0.0 else -1.0
        span = sign * clamp(abs(wedge),
                            math.radians(config.LPAD_SPAN_MIN),
                            math.radians(config.LPAD_SPAN_MAX))
        a0 = a_thumb + wedge * 0.5 - span * 0.5

        r0 = hand.size * config.LPAD_R0
        r1 = hand.size * config.LPAD_R1
        self._geom = (np.asarray(corner, np.float32), a0, span, r0, r1)
        self._draw_ring(ov, a, t, hand)

    # ------------------------------------------------------------------------
    @staticmethod
    def _sector(corner, a0, a1, r0, r1, steps=6):
        """Polygon outlining one annular slice: out along a0, back along a1."""
        angles = np.linspace(a0, a1, steps)
        inner = [(corner[0] + math.cos(v) * r0, corner[1] + math.sin(v) * r0)
                 for v in angles]
        outer = [(corner[0] + math.cos(v) * r1, corner[1] + math.sin(v) * r1)
                 for v in angles[::-1]]
        return np.asarray(inner + outer, np.float32)

    def _draw_ring(self, ov, a, t, hand):
        corner, a0, span, r0, r1 = self._geom
        n = self.palette.count
        # Grow outward as it opens, so the pad unfolds out of the corner.
        ease = smoothstep(0.0, 1.0, a)
        r0e = r0 * (0.55 + 0.45 * ease)
        r1e = r0e + (r1 - r0) * ease

        # Guides along the fan's two outer edges.
        for ang in (a0, a0 + span):
            d = np.array([math.cos(ang), math.sin(ang)], np.float32)
            ov.line(corner + d * (r0e * 0.55), corner + d * (r1e * 1.06),
                    config.WHITE, 0.22 * a, 1)
        ov.circle(corner, max(2.0, r0e * 0.14), config.WHITE, 0.5 * a, 1)

        seg = span / n
        gap = seg * 0.12
        for i in range(n):
            hot = (i == self.highlight)
            col = self.palette.swatch(i)
            # Each swatch is a filled annular slice, not a thick arc. An arc
            # wide enough to fill this band would be wider than a slice is
            # long, and neighbours would overlap into one smear.
            poly = self._sector(corner, a0 + seg * i + gap,
                                a0 + seg * (i + 1) - gap,
                                r0e, r1e + (7.0 if hot else 0.0))
            ov.fill_poly(poly, col, (0.46 if hot else 0.16) * a)
            ov.polyline(poly, col, (0.95 if hot else 0.36) * a,
                        2 if hot else 1, closed=True)
            if hot:
                ov.polyline(poly, config.WHITE, 0.30 * a, 1, closed=True)

        self._draw_marker(ov, corner, a0, span, r0e, r1e, a, t)
        self._draw_label(ov, corner, a0, span, r1e, a, hand)

        if self._pointer is not None:
            p = self._pointer
            ov.circle(p, 9.0 + 3.0 * math.sin(t * 7.0), config.WHITE, 0.7 * a, 1)
            ov.dot(p, 2.0, self.palette.swatch(self.highlight), 0.95 * a)
            if self._dwell > 0.0:
                k = clamp(self._dwell / max(config.LPAD_DWELL, 1e-3))
                ov.arc(p, 13.0, -90.0, -90.0 + 360.0 * k,
                       config.AMBER, 0.9 * a, 2)

    def _draw_marker(self, ov, corner, a0, span, r0e, r1e, a, t):
        """The needle sitting on the highlighted swatch."""
        n = self.palette.count
        k = (self.highlight + 0.5) / n
        ang = a0 + span * k
        d = np.array([math.cos(ang), math.sin(ang)], np.float32)
        col = self.palette.swatch(self.highlight)
        ov.line(corner + d * (r0e - 6.0), corner + d * (r1e + 8.0),
                col, 0.85 * a, 2)
        tip = corner + d * (r1e + 12.0)
        ov.dot(tip, 2.6, config.WHITE, 0.9 * a)
        flash = self._committed
        if flash > 0.01:
            ov.circle(tip, 8.0 + 26.0 * (1.0 - flash), config.WHITE,
                      flash ** 2 * 0.8, 1)

    def _draw_label(self, ov, corner, a0, span, r1e, a, hand):
        n = self.palette.count
        ang = a0 + span * 0.5
        d = np.array([math.cos(ang), math.sin(ang)], np.float32)
        anchor = corner + d * (r1e + 26.0)
        # Held inside the frame: the fan can sit anywhere, and a caption that
        # runs off the edge is worse than one nudged a few pixels.
        anchor = np.array([clamp(float(anchor[0]), 150.0, ov.w - 150.0),
                           clamp(float(anchor[1]), 44.0, ov.h - 60.0)],
                          np.float32)
        name = config.THEMES[self.highlight][0]
        tw, _ = ov.measure(name, 0.46)
        ov.text((anchor[0] - tw * 0.5, anchor[1]), name,
                self.palette.swatch(self.highlight), 0.95 * a, 0.46, 1,
                shift=True)
        sub = "ACCENT  %d/%d" % (self.highlight + 1, n)
        sw, _ = ov.measure(sub, 0.34)
        ov.text((anchor[0] - sw * 0.5, anchor[1] + 15), sub, config.DIM,
                0.7 * a, 0.34, 1)
        if hand is not None and self._pointer is None:
            hint = "TWIST TO DIAL  /  PINCH TO SET"
            hw, _ = ov.measure(hint, 0.30)
            ov.text((anchor[0] - hw * 0.5, anchor[1] + 30), hint, config.DIM,
                    0.45 * a, 0.30, 1)
