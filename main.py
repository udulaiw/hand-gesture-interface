"""Real-time hand tracking visual interface.

    python main.py

The webcam is the interface. Hands drive every overlay; there is no chrome
around the image.

Layered on the original tracking core are four instruments, each on its own
control axis so none of them can fight over a gesture:

    colour bar    two-hand pinch      grades the CAMERA
    accent pad    one-hand L frame    recolours the OVERLAY
    ORBIX viewer  unlock sequence     a holographic .glb model you tumble
    boot          on launch           the UDULAIW title sequence

Keys: ESC quits, H the gesture legend, D debug, F fullscreen, O the ORBIX
viewer, 1-9 pick a model, [ ] the accent, C X the camera filter, R reset.
"""
import argparse
import sys
import threading

import cv2
import numpy as np

import config
from effects.boot import BootSequence
from effects.color_filter import FilterBar
from effects.cursor import Cursor
from effects.hand_visualizer import HandVisualizer
from effects.holo import HoloViewer
from effects.hud import Hud
from effects.lpad import LPad
from effects.palm_grid import PalmGrid
from effects.particles import ParticleField
from effects.renderer import CameraGrade, Overlay
from effects.scan import ScanEffect
from effects.theme import ThemePalette
from effects.trails import TrailSystem
from tracking.camera import CameraError, CameraStream
from tracking.gestures import (FIST, LABELS, NONE, PEACE, THUMBS_UP,
                               GestureEngine)
from tracking.hand_tracker import HandTracker, ModelMissing, ensure_model
from utils.fps import FrameClock
from utils.smoothing import Ema

WINDOW = "HAND INTERFACE"

# While the boot sequence still owns most of the screen there is nothing to
# overlay onto, so the effect stack is skipped entirely and only the title
# draws. Below this veil the camera is visible enough for the interface to
# fade up over it.
_BOOT_HANDOVER = 0.55


class App:
    def __init__(self, args):
        self.args = args
        # Fail on a missing model before taking hold of the camera.
        ensure_model()
        # Opening the camera and loading the landmark model are both slow and
        # entirely independent, so they overlap. Media Foundation in particular
        # spends seconds negotiating a mode; there is no reason to spend them
        # waiting rather than building the tracker.
        opened = {}

        def open_camera():
            try:
                opened["camera"] = CameraStream(args.camera, args.width,
                                                args.height)
            except CameraError as exc:
                opened["error"] = exc

        thread = threading.Thread(target=open_camera, daemon=True)
        thread.start()
        self.tracker = HandTracker(config.MAX_HANDS)
        thread.join()
        if "error" in opened:
            raise opened["error"]
        self.camera = opened["camera"]
        self.w, self.h = self.camera.width, self.camera.height
        print(f"[camera] {self.w}x{self.h}  [{self.camera.mode}]")

        self.gestures = GestureEngine(config.MAX_HANDS)

        self.overlay = Overlay(self.w, self.h)
        self.grade = CameraGrade(self.w, self.h)
        self.hands_vis = HandVisualizer(config.MAX_HANDS)
        self.grid = PalmGrid(config.MAX_HANDS)
        self.particles = ParticleField(self.w, self.h)
        self.trails = TrailSystem(config.MAX_HANDS)
        self.cursor = Cursor(config.MAX_HANDS)
        self.scan = ScanEffect()
        self.filter_bar = FilterBar(self.w, self.h)
        if args.filter:
            self.filter_bar.select(args.filter)

        # The accent palette must be built before anything reads config.CYAN,
        # because selecting a theme rewrites that name in place.
        self.palette = ThemePalette(args.theme)
        self.lpad = LPad(self.palette)
        self.holo = HoloViewer(self.w, self.h)
        self.hud = Hud(self.w, self.h)

        self.clock = FrameClock()
        self.debug = args.debug
        # Self-test runs the real pipeline with no window, for diagnosing a
        # camera or measuring throughput without taking over the screen.
        self.headless = args.selftest > 0.0
        # Windowed by default. Everything is drawn in camera-frame coordinates
        # and scaled to whatever the window happens to be, so a small window is
        # a first-class way to run this, not a degraded one -- and taking over
        # the whole screen on launch is not a decision to make on the user's
        # behalf. F still toggles, and --fullscreen still starts that way.
        self.fullscreen = args.fullscreen and not self.headless
        self.quality = 1.0
        self.toggles = {"grid": True, "particles": True, "trails": True}
        self.help = Ema(0.14, 0.0)
        self.show_help = False
        self.running = True

        self.boot = None
        if not args.no_boot and not self.headless:
            self.boot = BootSequence(self.w, self.h)
        if args.model is not None:
            self.holo.select(args.model - 1)

        n = len(self.holo.library)
        print(f"[orbix] {n} model(s) available"
              + (f", {self.holo.library.missing} missing" if
                 self.holo.library.missing else ""))
        self._make_window()

    # ------------------------------------------------------------------ ui --
    def _make_window(self):
        if self.headless:
            return
        # KEEPRATIO holds the aspect while the window is resized, so hand
        # positions keep landing where the interface draws them however small
        # the window is dragged.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                        | cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        if not self.fullscreen:
            cv2.resizeWindow(WINDOW, self.args.window_width,
                             max(1, int(round(self.args.window_width
                                              * self.h / self.w))))

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        self.hud.toast("FULLSCREEN" if self.fullscreen else "WINDOWED")

    def _reset(self):
        self.particles.reset()
        self.trails.clear()
        self.cursor.reset()
        self.grid.reset()
        self.hands_vis.reset()
        self.filter_bar.reset()
        self.lpad.reset()
        self.holo.close()
        self.palette.select(0, instant=True)
        self.hud.toast("RESET")

    def _key(self, k):
        # Any key during the title sequence skips it, and does nothing else.
        if self.boot is not None and self.boot.running:
            self.boot.skip()
            return
        if k in (27, ord('q')):
            self.running = False
        elif k in (ord('d'), ord('D')):
            self.debug = not self.debug
            self.hud.toast("DEBUG " + ("ON" if self.debug else "OFF"))
        elif k in (ord('f'), ord('F')):
            self._toggle_fullscreen()
        elif k in (ord('h'), ord('H')):
            self.show_help = not self.show_help
        elif k in (ord('g'), ord('G')):
            self.toggles["grid"] = not self.toggles["grid"]
            self.hud.toast("PALM GRID " + ("ON" if self.toggles["grid"] else "OFF"))
        elif k in (ord('p'), ord('P')):
            self._toggle("particles", "PARTICLES")
        elif k in (ord('t'), ord('T')):
            self._toggle("trails", "TRAILS")
        elif k in (ord('c'), ord('C')):
            self.hud.toast("FILTER  " + self.filter_bar.cycle(1))
        elif k in (ord('x'), ord('X')):
            self.filter_bar.clear()
            self.hud.toast("FILTER OFF")
        elif k in (ord('o'), ord('O')):
            self.holo.toggle()
        elif k in (ord('n'), ord('N')):
            self.holo.step(1)
        elif k in (ord('b'), ord('B')):
            self.holo.step(-1)
        elif k in (ord('s'), ord('S')):
            if self.holo.open:
                self.holo.toggle_spin()
        elif k == ord('['):
            self.hud.toast("ACCENT  " + self.palette.cycle(-1))
        elif k == ord(']'):
            self.hud.toast("ACCENT  " + self.palette.cycle(1))
        elif ord('1') <= k <= ord('9'):
            self.holo.select(k - ord('1'))
        elif k in (ord('r'), ord('R')):
            self._reset()

    def _toggle(self, name, label):
        self.toggles[name] = not self.toggles[name]
        if name == "trails" and not self.toggles[name]:
            self.trails.clear()
        self.hud.toast(label + " " + ("ON" if self.toggles[name] else "OFF"))

    # -------------------------------------------------------------- events --
    def _handle_events(self, events, hands):
        for kind, idx, payload in events:
            hand = hands[idx]
            if kind == "pinch_start":
                # An index pinch is how the accent pad commits, so it gets
                # first refusal on the event before the particles react.
                if self.lpad.note_pinch(idx):
                    self.cursor.confirm(payload, self.palette.accent)
                    continue
                self.cursor.confirm(payload)
                if self.toggles["particles"]:
                    self.particles.implode(payload, 80)
                    self.particles.burst(payload, 40, 300)
            elif kind == "pinch_end":
                if self.toggles["particles"]:
                    self.particles.burst(payload, 46, 380)
            elif kind == "extra_pinch":
                self._extra_pinch(payload, hand)
            elif kind == "clench":
                self._clench(payload)
            elif kind == "unlock":
                self.holo.show()
                if self.toggles["particles"]:
                    self.particles.implode(
                        (self.w * 0.5, self.h * 0.5), 220)
            elif kind == "both_fists":
                if self.holo.open:
                    self.holo.close()
                elif self.show_help:
                    self.show_help = False
                else:
                    self.hud.toast("NOTHING TO DISMISS")
            elif kind == "wave":
                self.trails.clear_strokes()
                self.hud.toast("TRAILS CLEARED")
            elif kind == "gesture":
                if payload == THUMBS_UP:
                    self.cursor.confirm(hand.pts[4], config.CYAN)
                    self.hud.toast("CONFIRMED")
                elif payload == PEACE:
                    self.hud.toast("DRAW MODE")
                elif payload == FIST:
                    self.hud.toast("COLLAPSE")

    def _extra_pinch(self, finger, hand):
        """Thumb to middle / ring / pinky.

        The three of them mean different things depending on whether the ORBIX
        viewer holds the screen, which is the whole reason the viewer is behind
        an unlock: while it is closed these are interface toggles, and while it
        is open they drive the model. There is never a moment where one binding
        shadows the other.
        """
        if self.holo.open:
            if finger == "middle":
                self.holo.toggle_spin()
            elif finger == "ring":
                self.holo.step(-1)
            elif finger == "pinky":
                self.holo.step(1)
            return
        if finger == "middle":
            self.hud.toast("ACCENT  " + self.palette.cycle(1))
        elif finger == "ring":
            self._toggle("particles", "PARTICLES")
        elif finger == "pinky":
            self._toggle("trails", "TRAILS")

    def _clench(self, at):
        self.hud.toast("SHOCKWAVE")
        self.cursor.confirm(at, config.AMBER)
        if self.toggles["particles"]:
            self.particles.burst(at, 150, 700)

    # ---------------------------------------------------------------- loop --
    def run(self):
        stats = {"frames": 0, "with_hands": 0, "gestures": set()}
        seq = -1
        mirror = None
        while self.running:
            dt = self.clock.tick()
            t = self.clock.elapsed
            self._govern(dt)

            if not self.camera.alive:
                self._camera_lost()
                break

            raw, new_seq = self.camera.read_new(seq)
            if raw is None:
                if mirror is None:
                    continue
            else:
                seq = new_seq
                # Mirror so the interface behaves like a mirror, not a monitor.
                # Into a reused buffer: at 720p a fresh allocation per frame is
                # a megabyte and a half of churn for nothing.
                if mirror is None or mirror.shape != raw.shape:
                    mirror = np.empty_like(raw)
                cv2.flip(raw, 1, dst=mirror)
                # Only genuinely new frames are worth an inference. The render
                # loop runs two to three times faster than the camera delivers,
                # so without this gate most inferences re-derived landmarks the
                # tracker already had -- burning the budget that the next real
                # frame needed and adding latency to every gesture.
                self.tracker.submit(mirror)
            frame = mirror

            hands = self.tracker.poll(self.w, self.h, t, dt)
            events = self.gestures.update(hands, t, dt)
            self._handle_events(events, hands)

            veil = 0.0
            if self.boot is not None and self.boot.running:
                self.boot.update(dt)
                veil = self.boot.veil

            self.palette.update(dt)
            for kind, payload in self.lpad.update(hands, t, dt):
                if kind == "toast":
                    self.hud.toast(payload)
            for kind, payload in self.holo.update(hands, t, dt):
                if kind == "toast":
                    self.hud.toast(payload)

            # The viewer borrows the two-hand pinch for scaling while it is
            # open, so the colour bar stands down rather than both reacting.
            for kind, payload in self.filter_bar.update(
                    hands, t, dt, enabled=not self.holo.open):
                if kind == "filter_set":
                    self.hud.toast(payload)
                elif kind == "filter_arm" and self.toggles["particles"]:
                    for p in self.filter_bar.anchors:
                        self.particles.implode(p, 34)
            self.grade.set_filter(self.filter_bar.index,
                                  self.filter_bar.strength)

            ov = self.overlay
            ov.clear()
            if veil < _BOOT_HANDOVER:
                self._draw_effects(ov, hands, t, dt)
            if self.boot is not None and self.boot.running:
                self.boot.draw(ov, t)

            base = self.grade(frame)
            if veil > 0.002:
                cv2.convertScaleAbs(base, dst=base, alpha=1.0 - veil)
            out = ov.composite(base, self.quality)

            if self.headless:
                stats["frames"] += 1
                n = sum(1 for h in hands if h.visible)
                if n:
                    stats["with_hands"] += 1
                for h in hands:
                    if h.visible and h.gesture != NONE:
                        stats["gestures"].add(h.gesture)
                if t >= self.args.selftest:
                    self._report(stats, out)
                    self.running = False
                continue

            cv2.imshow(WINDOW, out)
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                self._key(k)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False

    def _govern(self, dt):
        """Trade effect detail for frame rate, with a wide hysteresis band.

        Quality rides between 0 and 1 and the effects read it as a hint. The
        two thresholds are far apart on purpose: a single band would leave the
        interface visibly pulsing between detail levels every time the frame
        time brushed the line.
        """
        fps = self.clock.fps
        if fps <= 0.0:
            return
        if fps < config.QUALITY_FPS_LOW:
            self.quality = max(0.0, self.quality - dt * 1.2)
        elif fps > config.QUALITY_FPS_HIGH:
            self.quality = min(1.0, self.quality + dt * 0.5)

    def _draw_effects(self, ov, hands, t, dt):
        if self.toggles["particles"]:
            self.particles.update(hands, t, dt)
            self.particles.draw(ov, t, self.quality)

        # The hologram sits behind the hands so they read as being in front of
        # it, which is what sells it as occupying the space between them.
        self.holo.draw(ov, hands, t, dt, self.quality)

        self.scan.draw(ov, hands, t, dt)
        if self.toggles["grid"]:
            self.grid.draw(ov, hands, t, dt)
        self.hands_vis.draw(ov, hands, t, dt)

        if self.toggles["trails"]:
            self.trails.update(hands, t, dt)
            self.trails.draw(ov, hands, t)

        self.cursor.update(dt)
        self.cursor.draw(ov, hands, t, dt)
        self.filter_bar.draw(ov, t)
        self.lpad.draw(ov, hands, t, dt)

        gname = self._headline(hands)
        self.hud.draw(ov, hands, self.clock, gname, self.toggles,
                      self.debug, self.tracker, self.camera)
        self.hud.draw_theme_chip(ov, self.palette)
        if not self.holo.open:
            self.hud.draw_unlock(ov, self.gestures.unlock_progress)
        self.hud.draw_dismiss(ov, self.gestures.both_fists_progress)
        a = self.help(1.0 if self.show_help else 0.0, dt)
        self.hud.draw_help(ov, a)

    def _report(self, stats, frame):
        f = stats["frames"]
        print(f"[selftest] {f} frames in {self.clock.elapsed:.1f}s  "
              f"-> {self.clock.fps:.1f} fps")
        print(f"[selftest] frames with a hand: {stats['with_hands']} "
              f"({100.0 * stats['with_hands'] / max(f, 1):.0f}%)")
        print(f"[selftest] inference latency: {self.tracker.latency_ms:.1f} ms")
        print(f"[selftest] inference rate: {self.tracker.infer_fps:.1f} hz  "
              f"(camera {self.camera.capture_fps:.1f} hz)")
        print(f"[selftest] gestures seen: "
              f"{sorted(stats['gestures']) or 'none'}")
        print(f"[selftest] orbix models: {len(self.holo.library)}")
        if self.args.capture:
            cv2.imwrite(self.args.capture, frame)
            print(f"[selftest] wrote {self.args.capture}")

    @staticmethod
    def _headline(hands):
        for hand in hands:
            if hand.visible and hand.gesture != NONE:
                return LABELS.get(hand.gesture, hand.gesture)
        return "-"

    def _camera_lost(self):
        ov = self.overlay
        ov.clear()
        black = np.zeros((self.h, self.w, 3), np.uint8)
        self.hud.draw_message(ov, [
            "CAMERA SIGNAL LOST",
            "reconnect the device and restart",
        ], config.RED)
        cv2.imshow(WINDOW, ov.composite(black))
        cv2.waitKey(2200)

    def close(self):
        self.camera.release()
        self.tracker.close()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Real-time hand tracking interface")
    p.add_argument("--camera", type=int, default=config.CAM_INDEX)
    p.add_argument("--width", type=int, default=config.CAM_WIDTH)
    p.add_argument("--height", type=int, default=config.CAM_HEIGHT)
    p.add_argument("--windowed", action="store_true",
                   help="start windowed (the default; kept for compatibility)")
    p.add_argument("--fullscreen", action="store_true",
                   help="start fullscreen instead of in a window")
    p.add_argument("--window-width", type=int, default=960, metavar="PX",
                   help="width of the initial window; the height follows the "
                        "camera's aspect ratio (default: 960)")
    p.add_argument("--debug", action="store_true", help="start with debug on")
    p.add_argument("--no-boot", action="store_true",
                   help="skip the UDULAIW title sequence")
    p.add_argument("--theme", type=int, default=0, metavar="N",
                   help="start on accent theme N (0-%d): %s"
                        % (len(config.THEMES) - 1,
                           ", ".join(t[0] for t in config.THEMES)))
    p.add_argument("--model", type=int, default=None, metavar="N",
                   help="open the ORBIX viewer on model N (1-%d)"
                        % len(config.ORBIX_MODELS))
    p.add_argument("--selftest", type=float, default=0.0, metavar="SECONDS",
                   help="run headless for N seconds and report camera, "
                        "throughput and tracking stats, then exit")
    p.add_argument("--filter", metavar="NAME", default=None,
                   help="start with a colour filter latched, e.g. THERMAL. "
                        "Names: " + ", ".join(f[0] for f in config.FILTERS))
    p.add_argument("--capture", metavar="PATH",
                   help="with --selftest, write the final composited frame here")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        app = App(args)
    except ModelMissing as e:
        print(f"\n[model] {e}\n", file=sys.stderr)
        return 2
    except CameraError as e:
        print(f"\n[camera] {e}\n", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"\n[filter] {e}\n", file=sys.stderr)
        return 4

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
