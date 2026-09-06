"""A mesh mapped onto the palm.

The grid is not drawn around the hand — it is drawn *in the palm's own plane*.
A quad is built from the wrist and the index/pinky knuckles, then each corner is
pushed toward or away from the quad centre according to its relative depth. That
turns the parallelogram into a keystoned trapezoid, and a homography from the
unit square onto it carries every grid vertex along. Rotation, scale and
perspective therefore come out of the geometry for free.
"""
import math

import cv2
import numpy as np

import config
from tracking import landmarks as L
from tracking.gestures import FIST, OPEN_PALM, PEACE
from utils.smoothing import Ema, clamp, smoothstep

# Palm-plane extents, in units of the wrist->knuckle vector.
_S0, _S1 = -0.60, 0.60
_T0, _T1 = -0.10, 1.08
_DEPTH_K = 0.62


class PalmGrid:
    def __init__(self, n_hands, cols=config.GRID_COLS, rows=config.GRID_ROWS):
        self.cols = cols
        self.rows = rows
        self._alpha = [Ema(0.16, 0.0) for _ in range(n_hands)]
        self._unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        self._sweep = [0.0] * n_hands

    def reset(self):
        for a in self._alpha:
            a.value = 0.0
        self._sweep = [0.0] * len(self._sweep)

    # ------------------------------------------------------------------------
    def draw(self, ov, hands, t, dt):
        for i, hand in enumerate(hands):
            want = 0.0
            if hand.visible and hand.gesture in (OPEN_PALM, PEACE):
                # Only show it when the palm actually faces the camera: a very
                # foreshortened palm has almost no area to map onto.
                facing = smoothstep(0.34, 0.62, self._facing(hand))
                want = facing * clamp(hand.openness * 1.3)
            elif hand.visible and hand.gesture == FIST:
                want = 0.0

            a = self._alpha[i](want, dt) * hand.presence
            if a < 0.012:
                continue
            self._sweep[i] = (self._sweep[i] + dt * 0.42) % 1.0
            self._draw_grid(ov, hand, a, self._sweep[i], t)

    @staticmethod
    def _facing(hand):
        """Palm area relative to hand size — a proxy for how square-on it is."""
        p = hand.pts
        across = np.linalg.norm(p[L.INDEX_MCP] - p[L.PINKY_MCP])
        return float(across) / max(hand.size, 1e-3)

    # ------------------------------------------------------------------------
    def _quad(self, hand):
        p = hand.pts
        w = p[L.WRIST].astype(np.float32)
        mid = ((p[L.INDEX_MCP] + p[L.PINKY_MCP]) * 0.5).astype(np.float32)
        along = mid - w
        across = (p[L.INDEX_MCP] - p[L.PINKY_MCP]).astype(np.float32)

        zw = float(hand.z[L.WRIST])
        zi = float(hand.z[L.INDEX_MCP])
        zp = float(hand.z[L.PINKY_MCP])
        zmid = 0.5 * (zi + zp)

        corners = []
        depths = []
        for s, tt in ((_S0, _T0), (_S1, _T0), (_S1, _T1), (_S0, _T1)):
            corners.append(w + along * tt + across * s)
            depths.append(zw + tt * (zmid - zw) + s * (zi - zp))
        corners = np.array(corners, np.float32)
        depths = np.array(depths, np.float32)

        centre = corners.mean(axis=0)
        zc = float(depths.mean())
        # Nearer corners expand, farther corners contract => keystone.
        rel = (depths - zc) / max(hand.size, 1e-3)
        scale = 1.0 / (1.0 + np.clip(rel, -0.8, 0.8) * _DEPTH_K)
        quad = centre + (corners - centre) * scale[:, None]
        return quad.astype(np.float32), centre

    def _draw_grid(self, ov, hand, alpha, sweep, t):
        quad, centre = self._quad(hand)
        try:
            H = cv2.getPerspectiveTransform(self._unit, quad)
        except cv2.error:
            return

        cols, rows = self.cols, self.rows
        us = np.linspace(0.0, 1.0, cols + 1, dtype=np.float32)
        vs = np.linspace(0.0, 1.0, rows + 1, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        pts = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H).reshape(rows + 1, cols + 1, 2)

        # Cull if the hand runs far off screen; nothing useful to draw.
        if not np.isfinite(mapped).all():
            return

        line_a = alpha * 0.30
        for r in range(rows + 1):
            edge = (r == 0 or r == rows)
            band = 1.0 - min(1.0, abs(r / rows - sweep) * 6.0)
            ov.polyline(mapped[r], config.WHITE,
                        line_a * (1.5 if edge else 1.0) + alpha * 0.45 * band, 1)
        for c in range(cols + 1):
            edge = (c == 0 or c == cols)
            ov.polyline(mapped[:, c], config.WHITE,
                        line_a * (1.5 if edge else 1.0), 1)

        # Intersection ticks, denser toward the palm centre.
        step = 2
        node_a = alpha * 0.5
        for r in range(1, rows, step):
            for c in range(1, cols, step):
                ov.dot(mapped[r, c], 1.0, config.CYAN, node_a)

        self._brackets(ov, quad, alpha)
        self._readout(ov, hand, quad, alpha)

    @staticmethod
    def _brackets(ov, quad, alpha):
        """Corner brackets with node circles, as in the reference HUD."""
        n = len(quad)
        for i in range(n):
            p = quad[i]
            a = quad[(i - 1) % n]
            b = quad[(i + 1) % n]
            for nb in (a, b):
                d = nb - p
                ln = float(np.linalg.norm(d))
                if ln < 1e-3:
                    continue
                d = d / ln
                ov.line(p, p + d * min(ln * 0.24, 26.0), config.AMBER,
                        alpha * 0.85, 1)
            ov.circle(p, 3.0, config.CYAN, alpha * 0.9, 1)

    @staticmethod
    def _readout(ov, hand, quad, alpha):
        if alpha < 0.35:
            return
        top = quad[np.argmin(quad[:, 1])]
        deg = int(round(math.degrees(hand.rotation) + 90)) % 360
        w = int(np.linalg.norm(quad[1] - quad[0]))
        h = int(np.linalg.norm(quad[3] - quad[0]))
        ov.text((top[0] + 8, top[1] - 8), f"PALM {w}x{h}  {deg:03d}",
                config.CYAN, alpha * 0.8, 0.36)
