"""The acquisition sweep played when a hand first enters frame.

A line travels along the hand's own long axis — wrist to fingertips — rather
than across the screen, so it reads as the hand being measured. Landmarks light
up as the line passes them and a bracket closes around the hand. It lasts about
a second and then gets out of the way.
"""
import numpy as np

import config
from tracking import landmarks as L
from utils.smoothing import clamp, smoothstep


class ScanEffect:
    def draw(self, ov, hands, t, dt):
        for hand in hands:
            if not hand.visible or hand.scan_t >= 1.0:
                continue
            self._draw(ov, hand)

    @staticmethod
    def _draw(ov, hand):
        k = hand.scan_t
        # Fade in fast, hold, fade out over the last third.
        a = smoothstep(0.0, 0.12, k) * (1.0 - smoothstep(0.72, 1.0, k))
        if a <= 0.01:
            return

        p = hand.pts
        along = p[L.MIDDLE_MCP] - p[L.WRIST]
        n = float(np.linalg.norm(along))
        if n < 1e-3:
            return
        along = along / n
        perp = np.array([-along[1], along[0]], np.float32)

        rel = p - p[L.WRIST]
        proj = rel @ along
        side = rel @ perp
        lo, hi = float(proj.min()), float(proj.max())
        # Clamp to the hand's own width; a splayed thumb would otherwise throw
        # the sweep line far past the silhouette.
        half = min(max(float(np.abs(side).max()), hand.size * 0.40),
                   hand.size * 0.80)

        # Ease the sweep so it decelerates into the fingertips.
        e = 1.0 - (1.0 - k) ** 2
        cut = lo + (hi - lo) * e
        centre = p[L.WRIST] + along * cut

        ov.line(centre - perp * half, centre + perp * half,
                config.CYAN, 0.60 * a, 1)
        c2 = p[L.WRIST] + along * (cut - 7.0)
        ov.line(c2 - perp * half, c2 + perp * half, config.CYAN, 0.18 * a, 1)
        ov.dot(centre + perp * half, 1.8, config.CYAN, 0.75 * a)
        ov.dot(centre - perp * half, 1.8, config.CYAN, 0.75 * a)

        # Landmarks flare as the line crosses them.
        band = max(hand.size * 0.30, 10.0)
        for i in range(21):
            d = abs(float(proj[i]) - cut)
            if d < band:
                lit = (1.0 - d / band) ** 2
                ov.dot(p[i], 1.4 + 1.8 * lit, config.WHITE, 0.9 * lit * a)
                ov.circle(p[i], 2.5 + 5.0 * lit, config.CYAN, 0.28 * lit * a, 1)

        ScanEffect._bracket(ov, hand, a, k)

    @staticmethod
    def _bracket(ov, hand, a, k):
        x0, y0, x1, y1 = hand.bbox
        pad = hand.size * (0.26 - 0.16 * clamp(k * 1.4))   # closes inward
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        arm = min(x1 - x0, y1 - y0) * 0.18
        corners = (
            ((x0, y0), (1, 0), (0, 1)),
            ((x1, y0), (-1, 0), (0, 1)),
            ((x1, y1), (-1, 0), (0, -1)),
            ((x0, y1), (1, 0), (0, -1)),
        )
        for (cx, cy), dx, dy in corners:
            ov.line((cx, cy), (cx + dx[0] * arm, cy + dx[1] * arm),
                    config.WHITE, 0.42 * a, 1)
            ov.line((cx, cy), (cx + dy[0] * arm, cy + dy[1] * arm),
                    config.WHITE, 0.42 * a, 1)

        label_a = smoothstep(0.25, 0.5, k) * (1.0 - smoothstep(0.75, 1.0, k))
        if label_a > 0.02:
            ov.text((x0, y0 - 8), "HAND DETECTED", config.WHITE,
                    label_a * 0.9, 0.38, 1, mono=True, shift=True)
