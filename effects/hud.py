"""The frame furniture: status line, corner ticks, centre reticle, debug panel.

Presentation mode keeps this to a handful of small captions at the edges. All
the numbers live behind D so the camera is never competing with a dashboard.
"""
import time

import numpy as np

import config
from tracking.gestures import LABELS


class Toast:
    __slots__ = ("text", "t", "life")

    def __init__(self, text, life=1.5):
        self.text = text
        self.t = 0.0
        self.life = life


class Hud:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self._toasts = []
        self._blink = 0.0

    def toast(self, text, life=1.5):
        self._toasts.append(Toast(text, life))
        if len(self._toasts) > 4:
            self._toasts.pop(0)

    def resize(self, width, height):
        self.w, self.h = width, height

    # ------------------------------------------------------------------------
    def draw(self, ov, hands, clock, gesture_name, toggles, debug, tracker,
             camera=None):
        self._frame_ticks(ov)
        self._reticle(ov)
        self._status(ov, hands, clock, gesture_name)
        self._toast_stack(ov, clock.dt)
        if debug:
            self._debug(ov, hands, clock, toggles, tracker, camera)

    # ------------------------------------------------------------------------
    def _frame_ticks(self, ov):
        m, arm, a = 22, 16, 0.30
        w, h = self.w, self.h
        for (cx, cy), sx, sy in (((m, m), 1, 1), ((w - m, m), -1, 1),
                                 ((w - m, h - m), -1, -1), ((m, h - m), 1, -1)):
            ov.line((cx, cy), (cx + sx * arm, cy), config.WHITE, a, 1)
            ov.line((cx, cy), (cx, cy + sy * arm), config.WHITE, a, 1)

    def _reticle(self, ov):
        cx, cy = self.w // 2, self.h // 2
        r, a = 7, 0.34
        ov.line((cx - r, cy), (cx - 2, cy), config.WHITE, a, 1)
        ov.line((cx + 2, cy), (cx + r, cy), config.WHITE, a, 1)
        ov.line((cx, cy - r), (cx, cy - 2), config.WHITE, a, 1)
        ov.line((cx, cy + 2), (cx, cy + r), config.WHITE, a, 1)

    def _status(self, ov, hands, clock, gesture_name):
        n = sum(1 for hand in hands if hand.visible)
        self._blink = (self._blink + clock.dt) % 1.6
        live = self._blink < 1.05

        # top-left tracking indicator
        if n:
            ov.dot((30, 30), 3.6, config.RED, 0.95 if live else 0.35)
            ov.text((42, 34), "TRACKING", config.WHITE, 0.9, 0.40, 1, shift=True)
        else:
            ov.circle((30, 30), 3.6, config.DIM, 0.5, 1)
            ov.text((42, 34), "STANDBY", config.DIM, 0.6, 0.40, 1)

        # top-right clock
        stamp = time.strftime("%d.%m.%Y  %H:%M:%S")
        tw, _ = ov.measure(stamp, 0.40)
        ov.text((self.w - tw - 28, 34), stamp, config.WHITE, 0.72, 0.40, 1,
                shift=True)

        # bottom-left status line
        parts = [f"HANDS {n}", gesture_name, f"FPS {clock.fps:04.1f}"]
        ov.text((28, self.h - 26), "  |  ".join(parts), config.WHITE, 0.85,
                0.42, 1, shift=True)

    def _toast_stack(self, ov, dt):
        for t in self._toasts:
            t.t += dt
        self._toasts = [t for t in self._toasts if t.t < t.life]
        y = self.h - 52
        for t in reversed(self._toasts):
            k = t.t / t.life
            a = min(1.0, (1.0 - k) * 3.0) * min(1.0, k * 8.0)
            ov.text((28, y), t.text, config.AMBER, a * 0.9, 0.38, 1)
            y -= 18

    # ------------------------------------------------------------------------
    def _debug(self, ov, hands, clock, toggles, tracker, camera=None):
        x, y = 28, 74
        lh = 17

        def row(text, color=config.CYAN, a=0.85):
            nonlocal y
            ov.text((x, y), text, color, a, 0.38, 1)
            y += lh

        ov.text((x, y), "-- DEBUG", config.AMBER, 0.9, 0.38, 1)
        y += lh + 3
        row(f"fps        {clock.fps:6.2f}   dt {clock.dt * 1000:5.1f} ms")
        # Render fps flatters the pipeline: it is the inference rate that
        # bounds how quickly a gesture can possibly be noticed, and the capture
        # rate that bounds the inference rate. All three belong side by side.
        row(f"infer      {tracker.latency_ms:6.1f} ms   "
            f"{tracker.infer_fps:5.1f} hz")
        row(f"frame      {self.w}x{self.h}   detect w={config.DETECT_WIDTH}")
        row(f"camera     {camera.mode}  {camera.capture_fps:.1f} hz"
            if camera is not None else "camera     -")
        row("toggles    " + " ".join(
            f"{k}:{'on' if v else 'off'}" for k, v in toggles.items()))
        y += 4

        for i, hand in enumerate(hands):
            if not hand.active:
                continue
            tip = hand.index_tip
            row(f"[{i}] {hand.label:<5} conf {hand.score:.2f}  "
                f"lost {hand.lost}", config.WHITE)
            row(f"    gesture   {LABELS.get(hand.gesture, hand.gesture)}")
            row(f"    index tip {tip[0]:7.1f}, {tip[1]:7.1f}")
            row(f"    size      {hand.size:6.1f} px   rot "
                f"{np.degrees(hand.rotation):7.1f} deg")
            row(f"    tilt      {hand.tilt[0]:+.2f}, {hand.tilt[1]:+.2f}   "
                f"speed {hand.speed:5.2f} hw/s")
            d = hand.pinch_dists
            row(f"    pinch     {hand.pinch_finger or '-':<6} "
                f"conf {hand.pinch_conf:.2f}  on<{config.PINCH_ON}")
            row(f"    tip dist  i {hand.pinch_dist:.2f}  m {d[0]:.2f}  "
                f"r {d[1]:.2f}  p {d[2]:.2f}")
            ext = "  ".join(f"{v:.2f}" for v in hand.extension)
            row(f"    extension {ext}")
            y += 3

        # Live landmark scatter for the first visible hand.
        hand = next((h for h in hands if h.visible), None)
        if hand is not None:
            for p in hand.pts:
                ov.dot(p, 1.0, config.AMBER, 0.55)
            ov.polyline([hand.bbox[:2], (hand.bbox[2], hand.bbox[1]),
                         hand.bbox[2:], (hand.bbox[0], hand.bbox[3])],
                        config.AMBER, 0.30, 1, closed=True)

    # ------------------------------------------------- ORBIX extension panels
    def draw_unlock(self, ov, progress, hot=False):
        """Pips showing how far through the ORBIX unlock sequence we are.

        Drawn only while the sequence is actually part-way done, so it appears
        exactly when it is useful -- confirming the first pose registered --
        and is invisible the rest of the time.
        """
        n = len(config.UNLOCK_SEQUENCE)
        step = int(round(progress * n))
        if step <= 0 and not hot:
            return
        x = self.w * 0.5 - (n - 1) * 9.0
        y = 30.0
        for i in range(n):
            done = i < step
            ov.circle((x + i * 18.0, y), 4.0,
                      config.AMBER if done else config.DIM,
                      0.95 if done else 0.30, -1 if done else 1)
        label = config.UNLOCK_SEQUENCE[min(step, n - 1)].replace("_", " ").upper()
        tw, _ = ov.measure(label, 0.32)
        ov.text((self.w * 0.5 - tw * 0.5, y + 18), label, config.DIM, 0.6,
                0.32, 1)

    def draw_dismiss(self, ov, progress):
        """Closing ring for the two-fist dismiss, so the hold reads as timed."""
        if progress <= 0.02 or progress >= 1.0:
            return
        c = (self.w * 0.5, self.h * 0.5)
        ov.arc(c, 26.0, -90.0, -90.0 + 360.0 * progress, config.RED, 0.8, 2)
        ov.text((self.w * 0.5 + 34, self.h * 0.5 + 4), "DISMISS", config.RED,
                0.7, 0.34, 1)

    def draw_theme_chip(self, ov, palette):
        """Bottom-right marker for the live accent, mirroring the filter chip."""
        text = "ACC " + palette.name
        tw, _ = ov.measure(text, 0.40)
        x, y = self.w - tw - 28, self.h - 46
        ov.dot((x - 12, y - 4), 3.0, palette.accent, 0.9)
        ov.text((x, y), text, config.WHITE, 0.55 + 0.4 * palette.flash, 0.40, 1)

    def draw_help(self, ov, alpha=1.0):
        """The gesture legend. Two columns, drawn over a dimmed frame."""
        if alpha <= 0.02:
            return
        rows = (
            ("CORE", None),
            ("pinch  thumb+index", "grab / gather particles"),
            ("pinch  both hands", "colour bar  (length=filter, tilt=strength)"),
            ("open palm", "palm mesh + particle orbit"),
            ("two fingers", "draw mode"),
            ("point", "cursor reticle"),
            ("thumbs up", "confirm"),
            ("wave", "clear strokes"),
            ("fist", "collapse"),
            ("EXTENDED", None),
            ("L frame  thumb+index", "accent pad — twist to dial, pinch to set"),
            ("thumb + middle", "cycle accent / hold spin in ORBIX"),
            ("thumb + ring", "toggle particles / previous model"),
            ("thumb + pinky", "toggle trails / next model"),
            ("fist  (snap shut)", "shockwave collapse"),
            ("ORBIX", None),
            ("fist > palm > peace", "unlock the holographic viewer"),
            ("fist + move", "tumble the model"),
            ("pinch both hands", "scale the model"),
            ("both fists, hold", "dismiss"),
        )
        # Measure the block before drawing it: section headers each add their
        # own leading, so a panel sized from the row count alone comes out
        # short and the last bindings print outside it.
        headers = sum(1 for _, right in rows if right is None)
        title_h, gap_h, row_h, hint_h = 30, 8, 16, 30
        body = title_h + len(rows) * row_h + headers * gap_h + hint_h
        width = 640
        pad = 26
        x0 = self.w * 0.5 - width * 0.5 + pad
        top = self.h * 0.5 - body * 0.5
        right_edge = x0 + width - pad * 2

        ov.fill_poly([(x0 - pad, top), (right_edge + pad, top),
                      (right_edge + pad, top + body), (x0 - pad, top + body)],
                     config.INK, 0.75 * alpha)
        ov.polyline([(x0 - pad, top), (right_edge + pad, top),
                     (right_edge + pad, top + body), (x0 - pad, top + body)],
                    config.DIM, 0.30 * alpha, 1, closed=True)

        y = top + 20
        ov.text((x0, y), "GESTURE VOCABULARY", config.WHITE, 0.95 * alpha,
                0.46, 1, shift=True)
        y += title_h - 8
        for left, right in rows:
            if right is None:
                y += gap_h
                ov.text((x0, y), left, config.AMBER, 0.85 * alpha, 0.36, 1)
                tw, _ = ov.measure(left, 0.36)
                ov.line((x0 + tw + 12, y - 4), (right_edge, y - 4), config.DIM,
                        0.25 * alpha, 1)
            else:
                ov.text((x0 + 10, y), left, config.CYAN, 0.85 * alpha, 0.34, 1)
                ov.text((x0 + 230, y), right, config.WHITE, 0.62 * alpha,
                        0.34, 1)
            y += row_h
        ov.text((x0, y + 14), "H closes this   |   ESC quits", config.DIM,
                0.6 * alpha, 0.32, 1)

    # ------------------------------------------------------------------------
    def draw_message(self, ov, lines, color=config.WHITE):
        """Centred block, used for the camera-failure screen."""
        y = self.h // 2 - len(lines) * 12
        for i, line in enumerate(lines):
            a = 0.95 if i == 0 else 0.6
            tw, _ = ov.measure(line, 0.46)
            ov.text(((self.w - tw) // 2, y), line, color, a, 0.46, 1)
            y += 26
