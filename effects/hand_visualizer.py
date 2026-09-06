"""Custom hand rendering — nothing from MediaPipe's default drawing utils.

Thin tapered bones, small hollow joint nodes, fingertip rings that respond to
extension, and an overall intensity driven by movement. A fist contracts the
whole skeleton toward the palm; that collapse is animated, not switched.
"""
import math

import numpy as np

import config
from tracking import landmarks as L
from tracking.gestures import FIST
from utils.smoothing import Ema, clamp, smoothstep


class _Anim:
    def __init__(self):
        self.collapse = Ema(0.13, 0.0)
        self.intensity = Ema(0.10, 0.0)
        self.tip = [Ema(0.09, 0.0) for _ in range(5)]


class HandVisualizer:
    def __init__(self, n_hands):
        self._anim = [_Anim() for _ in range(n_hands)]

    def reset(self):
        for a in self._anim:
            a.collapse.value = 0.0
            a.intensity.value = 0.0

    def draw(self, ov, hands, t, dt):
        for i, hand in enumerate(hands):
            if not hand.active:
                continue
            self._draw_hand(ov, hand, self._anim[i], t, dt)

    # ------------------------------------------------------------------------
    def _draw_hand(self, ov, hand, anim, t, dt):
        a = anim
        collapse = a.collapse(1.0 if hand.gesture == FIST else 0.0, dt)
        motion = clamp(hand.speed / 3.2)
        intensity = a.intensity(0.55 + 0.45 * motion, dt)

        scan = smoothstep(0.0, 0.55, hand.scan_t)
        base = hand.presence * scan
        if base <= 0.01:
            return

        pts = hand.pts
        if collapse > 0.002:
            pts = pts + (hand.palm_center - pts) * (0.42 * collapse)

        s = hand.size
        pulse = 0.5 + 0.5 * math.sin(t * 2.1 + hand.rotation)
        glow = base * intensity

        # Palm plate — a very low-alpha fill anchors the hand without hiding it.
        palm = pts[list(L.PALM_LOOP)]
        ov.fill_poly(palm, config.CYAN, 0.032 * base * (1.0 - 0.6 * collapse))
        ov.polyline(palm, config.WHITE, 0.20 * base, 1, closed=True)

        # Bones: alpha ramps along each finger so tips read brightest.
        for chain in L.FINGERS:
            n = len(chain) - 1
            for k, (ia, ib) in enumerate(zip(chain[:-1], chain[1:])):
                f = k / max(n - 1, 1)
                ov.line(pts[ia], pts[ib], config.WHITE,
                        (0.30 + 0.42 * f) * glow, 1)
        for ia, ib in ((L.INDEX_MCP, L.MIDDLE_MCP), (L.MIDDLE_MCP, L.RING_MCP),
                       (L.RING_MCP, L.PINKY_MCP)):
            ov.line(pts[ia], pts[ib], config.WHITE, 0.22 * glow, 1)

        # Joint nodes
        r_node = max(1.2, s * 0.030)
        for idx in (L.THUMB_MCP, L.THUMB_IP, L.INDEX_PIP, L.INDEX_DIP,
                    L.MIDDLE_PIP, L.MIDDLE_DIP, L.RING_PIP, L.RING_DIP,
                    L.PINKY_PIP, L.PINKY_DIP):
            ov.dot(pts[idx], r_node * 0.55, config.WHITE, 0.34 * glow)
        for idx in L.MCPS:
            ov.circle(pts[idx], r_node, config.CYAN, 0.42 * glow, 1)

        # Wrist anchor — a small bracket rather than another dot.
        self._wrist_anchor(ov, hand, pts, 0.5 * base, s)

        # Fingertips
        for fi, tip_idx in enumerate(L.TIPS):
            e = float(hand.extension[fi])
            lit = a.tip[fi](e, dt)
            if lit < 0.04:
                continue
            p = pts[tip_idx]
            rr = max(1.4, s * 0.040) * (0.75 + 0.25 * lit)
            col = config.AMBER if lit > 0.72 else config.WHITE
            ov.dot(p, max(0.9, rr * 0.26), col,
                   (0.45 + 0.35 * pulse) * lit * base)
            ov.circle(p, rr, col, 0.26 * lit * base, 1)

    @staticmethod
    def _wrist_anchor(ov, hand, pts, alpha, s):
        w = pts[L.WRIST]
        ang = hand.rotation
        c, sn = math.cos(ang), math.sin(ang)
        # axis across the wrist, perpendicular to the palm's long axis
        ax = np.array([-sn, c], np.float32) * (s * 0.34)
        ov.line(w - ax, w + ax, config.WHITE, 0.30 * alpha, 1)
        for sgn in (-1.0, 1.0):
            e = w + ax * sgn
            back = np.array([c, sn], np.float32) * (s * 0.10)
            ov.line(e, e - back, config.WHITE, 0.30 * alpha, 1)
        ov.circle(w, max(1.5, s * 0.045), config.WHITE, 0.36 * alpha, 1)
