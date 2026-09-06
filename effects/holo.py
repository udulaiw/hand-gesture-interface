"""The ORBIX holographic model viewer.

A software wireframe renderer for the .glb models in assets/orbix, drawn onto
the same overlay canvas as everything else so it picks up the bloom pass and
sits in the same visual language as the rest of the interface.

The pipeline per frame is four vectorised NumPy steps and then a handful of
batched OpenCV calls:

  1. rotate every vertex by the current yaw / pitch / roll
  2. project with a real perspective divide, so the near face of a planet
     genuinely swells toward the camera
  3. cull back faces from the sign of each projected triangle's area -- which
     sign that is gets decided per frame from the depths of the two halves,
     because the models do not agree on winding order
  4. depth-sort what survives, keep the nearest N, and draw it in bands, each
     band at its own alpha so the far side of the mesh recedes

Step 4 is also the performance governor. `HOLO_MS_BUDGET` is a wall-clock
budget for the wireframe pass; if a frame overruns, N comes down, and the model
loses depth rather than the interface losing frame rate. That is what lets a
dense mesh and a slow machine coexist.

Control is entirely by hand:

    grab (fist) and move     tumble the model, with inertia on release
    two-hand pinch, spread   scale it
    thumb + pinky            next model          thumb + ring    previous
    thumb + middle           toggle the idle spin
    point                    range-find: a probe line onto the hull
    both fists, held         dismiss

The viewer only exists after the unlock sequence in `tracking/gestures.py`
fires, so none of these bindings can be triggered by accident while it is
closed.
"""
import math
import os
import threading
import time

import numpy as np

import config
from tracking.gestures import FIST, POINT
from utils.glb import GlbError, load_cached
from utils.smoothing import Ema, clamp, smoothstep


def _rotation(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], np.float32)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], np.float32)
    return (ry @ rx @ rz).astype(np.float32)


class ModelLibrary:
    """Lazily loads the configured .glb models, and remembers failures.

    Nothing is parsed until a model is actually shown. The first display of a
    model costs one parse (tens of milliseconds, once ever -- `load_cached`
    writes an .npz beside it); every later one is a cache read. A model that
    fails to load is recorded and skipped rather than retried each time it comes
    round in the carousel.
    """

    def __init__(self):
        self.entries = []
        for stem, label, accent in config.ORBIX_MODELS:
            path = os.path.join(config.ORBIX_DIR, stem + ".glb")
            if os.path.exists(path):
                self.entries.append({"path": path, "label": label,
                                     "accent": accent, "mesh": None,
                                     "error": None})
        self.missing = len(config.ORBIX_MODELS) - len(self.entries)
        self._loading = False

    def __len__(self):
        return len(self.entries)

    @property
    def available(self):
        return bool(self.entries)

    def get(self, index):
        """The entry at `index`, loading it on first use. Never raises."""
        if not self.entries:
            return None
        e = self.entries[index % len(self.entries)]
        if e["mesh"] is None and e["error"] is None:
            try:
                mesh = load_cached(e["path"], config.ORBIX_CACHE,
                                   config.ORBIX_MAX_TRIS)
            except (GlbError, OSError, ValueError, KeyError) as exc:
                e["error"] = str(exc)
                return e
            e["mesh"] = mesh
            # Fold the material's average hue into the configured accent, so a
            # multi-material model like the rocket keeps some of its own
            # colour instead of coming out flat.
            tint = np.clip(mesh.face_color.mean(axis=0), 0.55, 1.0)
            e["accent"] = tuple(float(c) * float(k)
                                for c, k in zip(e["accent"], tint))
        return e

    def label(self, index):
        if not self.entries:
            return "-"
        return self.entries[index % len(self.entries)]["label"]

    def prefetch(self, index, span=1):
        """Parse the models either side of `index` on a background thread.

        Every model is cached after its first parse, but the first parse still
        has to happen somewhere, and the frame it happened on was always the
        frame the user had just gestured on -- so stepping the carousel felt
        like it stuttered exactly once per model. Doing it ahead of time, off
        the render thread, moves that cost somewhere nobody is watching.
        """
        if not self.entries or not config.HOLO_PREFETCH:
            return
        n = len(self.entries)
        wanted = [(index + d) % n for d in range(-span, span + 1)]
        cold = [i for i in wanted if self.entries[i]["mesh"] is None
                and self.entries[i]["error"] is None]
        if not cold or self._loading:
            return
        self._loading = True

        def work():
            try:
                for i in cold:
                    self.get(i)
            finally:
                self._loading = False

        threading.Thread(target=work, daemon=True).start()


class HoloViewer:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.library = ModelLibrary()
        self.index = 0
        self.open = False
        self._open_ema = Ema(config.HOLO_OPEN, 0.0)

        self.yaw = 0.6
        self.pitch = -0.25
        self.roll = 0.0
        self.spin_yaw = 0.0
        self.spin_pitch = 0.0
        self.auto_spin = True
        self.scale = 1.0

        self._grab_hand = None
        self._grab_prev = None
        self._pinch_ref = None          # (gap, scale) when two-hand scaling armed
        self._centre = None
        self._centre_vel = np.zeros(2, np.float32)
        # The scale the gesture asks for, and the scale actually drawn. Keeping
        # them apart is what stops a jittery frame in the two-hand pinch from
        # arriving as a visible pop in the model's size.
        self.scale_shown = 1.0
        self._swap = 0.0
        self._swap_from = None          # index we are dissolving away from
        self._budget = float(config.HOLO_TRI_BUDGET)
        self._ms = 0.0
        self._drawn = 0
        self._probe = None
        self.events = []
        # Parse the first model now rather than on the frame the viewer opens.
        # A cold parse is tens of milliseconds; paying it during startup is
        # invisible, paying it mid-gesture is a dropped frame right at the
        # moment the user is watching for a response.
        self.library.get(self.index)
        # ...and warm its neighbours in the background, so the first carousel
        # step is as cheap as the fifteenth.
        self.library.prefetch(self.index)

    # ------------------------------------------------------------------------
    @property
    def visible(self):
        return float(self._open_ema.value or 0.0) > 0.008

    @property
    def label(self):
        return self.library.label(self.index)

    @property
    def available(self):
        return self.library.available

    def resize(self, width, height):
        self.w, self.h = width, height

    def toggle(self):
        return self.close() if self.open else self.show()

    def show(self):
        if not self.library.available:
            self.events.append(("toast", "NO ORBIX MODELS FOUND"))
            return False
        self.open = True
        self._swap_from = self.index
        self._swap = 1.0
        entry = self.library.get(self.index)
        if entry is not None and entry["error"]:
            self.events.append(("toast", "MODEL ERROR  " + self.label))
        else:
            self.events.append(("toast", "ORBIX  " + self.label))
        return True

    def close(self):
        if not self.open:
            return False
        self.open = False
        self._grab_hand = None
        self._grab_prev = None
        self._pinch_ref = None
        self.events.append(("toast", "ORBIX CLOSED"))
        return True

    def step(self, delta):
        """Advance the carousel. Silently does nothing while closed."""
        if not self.open or not self.library.available:
            return None
        self._swap_from = self.index
        self.index = (self.index + delta) % len(self.library)
        self._swap = 1.0
        self.library.get(self.index)
        self.library.prefetch(self.index)
        self.events.append(("toast", "ORBIX  " + self.label))
        return self.label

    def select(self, index):
        if not self.library.available:
            return None
        self._swap_from = self.index
        self.index = int(index) % len(self.library)
        self._swap = 1.0
        if not self.open:
            self.show()
        else:
            self.events.append(("toast", "ORBIX  " + self.label))
        return self.label

    def toggle_spin(self):
        self.auto_spin = not self.auto_spin
        self.events.append(("toast", "SPIN " + ("ON" if self.auto_spin else "OFF")))

    # ---------------------------------------------------------------- update --
    def update(self, hands, t, dt):
        """Fold hand motion into the transform. Returns queued app events."""
        out = list(self.events)
        self.events.clear()
        self._open_ema(1.0 if self.open else 0.0, dt)
        self._swap = max(0.0, self._swap - dt / max(config.HOLO_SWAP_TIME, 1e-3))
        if self._swap <= 0.0:
            self._swap_from = None
        # The drawn scale chases the gesture's scale rather than tracking it
        # exactly, so scaling reads as weighty instead of twitchy.
        a = 1.0 - math.exp(-math.log(2.0) / max(config.HOLO_SCALE_SMOOTH, 1e-3)
                           * max(dt, 1e-4))
        self.scale_shown += (self.scale - self.scale_shown) * a
        if not self.visible:
            return out
        if self.open:
            self.library.prefetch(self.index)

        self._track_centre(hands, dt)
        if self.open:
            self._track_grab(hands, dt)
            self._track_scale(hands)
            self._track_probe(hands)
        else:
            self._grab_hand = None
            self._probe = None

        if self._grab_hand is None:
            # Inertia from the last throw, decaying into the idle spin.
            damp = math.exp(-2.4 * dt)
            self.spin_yaw *= damp
            self.spin_pitch *= damp
            self.yaw += self.spin_yaw * dt
            self.pitch += self.spin_pitch * dt
            if self.auto_spin:
                self.yaw += config.HOLO_SPIN * dt
        self.pitch = clamp(self.pitch, -1.35, 1.35)
        self.yaw %= 2.0 * math.pi
        return out

    def _track_centre(self, hands, dt):
        cx, cy = self.w * 0.5, self.h * 0.5
        live = [h for h in hands if h.visible]
        if live:
            mid = np.mean([h.palm_center for h in live], axis=0)
            tx = cx + (float(mid[0]) - cx) * config.HOLO_FOLLOW
            ty = cy + (float(mid[1]) - cy) * config.HOLO_FOLLOW
        else:
            tx, ty = cx, cy
        target = np.array([tx, ty], np.float32)
        if self._centre is None:
            self._centre = target
            return
        # A critically damped spring rather than an exponential ease. The ease
        # is a single rate: slow enough to look smooth when the hand drifts and
        # it visibly trails a hand that sweeps, fast enough to keep up with a
        # sweep and it transmits every frame of landmark jitter while the hand
        # is still. A spring closes a large gap quickly and settles a small one
        # gently, which is the behaviour both cases wanted.
        h = min(max(dt, 1e-4), 1.0 / 30.0)
        k = float(config.HOLO_FOLLOW_STIFF)
        steps = max(1, int(math.ceil(max(dt, 1e-4) / h)))
        step = max(dt, 1e-4) / steps
        for _ in range(steps):
            accel = (target - self._centre) * k - self._centre_vel * (2.0 * math.sqrt(k))
            self._centre_vel += accel * step
            self._centre += self._centre_vel * step

    def _track_grab(self, hands, dt):
        """A closed fist tumbles the model; releasing it throws."""
        holder = None
        for i, hand in enumerate(hands):
            if hand.visible and hand.gesture == FIST:
                if self._grab_hand in (None, i):
                    holder = (i, hand)
                    break
        if holder is None:
            self._grab_hand = None
            self._grab_prev = None
            return

        i, hand = holder
        pos = np.asarray(hand.palm_center, np.float32)
        if self._grab_hand != i or self._grab_prev is None:
            self._grab_hand = i
            self._grab_prev = pos.copy()
            return

        radius = max(self._radius(), 1.0)
        d = (pos - self._grab_prev) / radius
        self._grab_prev = pos.copy()
        if dt > 1e-4:
            dy = float(d[0]) * config.HOLO_DRAG
            dp = float(d[1]) * config.HOLO_DRAG
            self.yaw += dy
            self.pitch += dp
            # Blend into the throw velocity rather than overwriting it, so a
            # single noisy frame at the moment of release cannot fling it.
            self.spin_yaw += (dy / dt - self.spin_yaw) * 0.35
            self.spin_pitch += (dp / dt - self.spin_pitch) * 0.35

    def _track_scale(self, hands):
        """Two index pinches: the gap between them scales the model."""
        pinched = [h for h in hands if h.visible and h.pinch]
        if len(pinched) < 2:
            self._pinch_ref = None
            return
        a = np.asarray(pinched[0].pinch_point, np.float32)
        b = np.asarray(pinched[1].pinch_point, np.float32)
        gap = float(np.hypot(*(b - a)))
        if gap < 12.0:
            return
        if self._pinch_ref is None:
            self._pinch_ref = (gap, self.scale)
            return
        gap0, scale0 = self._pinch_ref
        self.scale = clamp(scale0 * gap / max(gap0, 1e-3),
                           config.HOLO_SCALE_MIN, config.HOLO_SCALE_MAX)

    def _track_probe(self, hands):
        """A pointing finger casts a range-finder line onto the hull."""
        self._probe = None
        for hand in hands:
            if hand.visible and hand.gesture == POINT:
                self._probe = np.asarray(hand.index_tip, np.float32)
                return

    def _radius(self):
        base = min(self.w, self.h) * config.HOLO_RADIUS
        ease = smoothstep(0.0, 1.0, float(self._open_ema.value or 0.0))
        return base * self.scale_shown * (0.35 + 0.65 * ease)

    # ------------------------------------------------------------------ draw --
    def draw(self, ov, hands, t, dt, quality=1.0):
        a = float(self._open_ema.value or 0.0)
        if a < 0.008:
            return
        entry = self.library.get(self.index)
        if entry is None:
            return
        if entry["error"]:
            self._draw_error(ov, entry, a)
            return
        mesh = entry["mesh"]
        if mesh is None:
            return

        centre = self._centre if self._centre is not None else np.array(
            [self.w * 0.5, self.h * 0.5], np.float32)
        radius = self._radius()
        # The swap dissolve pinches the model in and flares a ring outward.
        # Eased rather than linear: a linear fade reaches full opacity while it
        # is still visibly moving, which is what made switching models look
        # like a snap with a fade bolted on rather than one motion.
        swap = smoothstep(0.0, 1.0, self._swap)
        alpha = a * (1.0 - 0.75 * swap)
        radius *= 1.0 - 0.22 * swap

        t0 = time.perf_counter()
        drawn = self._draw_mesh(ov, mesh, entry["accent"], centre, radius,
                                alpha, quality)
        self._govern((time.perf_counter() - t0) * 1000.0)
        self._drawn = drawn

        self._draw_frame(ov, centre, radius, alpha, t)
        if swap > 0.01:
            ov.circle(centre, radius * (1.0 + 1.6 * (1.0 - swap)),
                      config.WHITE, swap ** 2 * 0.55, 1)
        if self._probe is not None and self.open:
            self._draw_probe(ov, centre, radius, alpha, t)
        self._draw_panel(ov, mesh, entry, alpha)
        self._draw_carousel(ov, alpha)

    # ------------------------------------------------------------------------
    def _draw_mesh(self, ov, mesh, accent, centre, radius, alpha, quality=1.0):
        V, F = mesh.verts, mesh.faces
        p = V @ _rotation(self.yaw, self.pitch, self.roll).T
        z = p[:, 2]

        # Perspective divide. The eye sits HOLO_FOV radii out on +z, so a
        # vertex at z = 1 is genuinely nearer and draws larger.
        k = config.HOLO_FOV / np.maximum(config.HOLO_FOV - z, 0.35)
        scr = np.empty((len(V), 2), np.float32)
        scr[:, 0] = centre[0] + p[:, 0] * k * radius
        scr[:, 1] = centre[1] - p[:, 1] * k * radius

        tri = scr[F]
        ax = tri[:, 1, 0] - tri[:, 0, 0]
        ay = tri[:, 1, 1] - tri[:, 0, 1]
        bx = tri[:, 2, 0] - tri[:, 0, 0]
        by = tri[:, 2, 1] - tri[:, 0, 1]
        area = ax * by - ay * bx
        depth = z[F].mean(axis=1)

        # Winding order is not consistent across these exports, so rather than
        # trusting it, keep whichever half is genuinely closer to the camera.
        pos = area > 0.0
        if pos.any() and (~pos).any():
            front = pos if depth[pos].mean() > depth[~pos].mean() else ~pos
        else:
            front = np.ones(len(F), bool)

        tri = tri[front]
        depth = depth[front]
        if len(tri) == 0:
            return 0

        order = np.argsort(depth)           # far to near
        budget = max(int(self._budget), config.HOLO_TRI_FLOOR)
        if len(order) > budget:
            order = order[-budget:]         # keep the nearest, drop the far side
        tri = np.round(tri[order]).astype(np.int32)

        n = len(tri)
        bands = max(1, int(config.HOLO_BANDS))
        step = max(1, n // bands)
        for b in range(bands):
            lo = b * step
            hi = n if b == bands - 1 else min(n, lo + step)
            if hi <= lo:
                continue
            k_band = b / max(bands - 1, 1)
            # Far bands are nearly transparent so the mesh reads as a
            # hollow shell rather than a solid ball of lines.
            band_a = (0.09 + 0.46 * k_band) * alpha
            # Antialiasing only ever bought anything on the near bands, and
            # under load it is the first thing worth giving up: the bloom pass
            # softens the stair-stepping either way.
            ov.polys(tri[lo:hi], accent, band_a, 1, closed=True,
                     antialias=quality > 0.5 and b >= bands - 2)

        if quality > 0.2:
            self._draw_dust(ov, scr, z, accent, alpha)
        return n

    def _draw_dust(self, ov, scr, z, accent, alpha):
        """Vertex point cloud — what makes it read as volume, not just line art."""
        if config.HOLO_DUST <= 0.01:
            return
        inten = ((z + 1.0) * 0.5) ** 2.0 * config.HOLO_DUST * alpha
        ov.splat(scr, inten.astype(np.float32), config.WHITE)

    def _govern(self, ms):
        """Hold the wireframe pass to HOLO_MS_BUDGET by trading away triangles."""
        self._ms += (ms - self._ms) * 0.25
        budget = float(config.HOLO_MS_BUDGET)
        if self._ms > budget * 1.15:
            self._budget *= 0.94
        elif self._ms < budget * 0.70:
            self._budget *= 1.03
        self._budget = float(np.clip(self._budget, config.HOLO_TRI_FLOOR,
                                     config.HOLO_TRI_BUDGET))

    # ------------------------------------------------------------------------
    def _draw_frame(self, ov, centre, radius, alpha, t):
        """Orbit rings, axis pins and the containment brackets."""
        # Two rings in the model's own equatorial plane, flattened by the
        # pitch so they lean with it. A minimum height keeps an edge-on ring
        # from collapsing into a bare line.
        for rr, aa, phase in ((1.16, 0.34, 0.0), (1.34, 0.18, 2.0)):
            ry = max(2.0, abs(radius * rr * math.sin(self.pitch)) + 1.5)
            self._ellipse(ov, centre, radius * rr, ry, config.CYAN, aa * alpha)
            # A bead running the ring gives the hologram a heartbeat.
            ang = t * (0.9 + phase * 0.2) + phase
            ov.dot((centre[0] + math.cos(ang) * radius * rr,
                    centre[1] + math.sin(ang) * ry),
                   2.0, config.AMBER, 0.75 * alpha)

        # Polar axis.
        up = radius * 1.30
        ov.line((centre[0], centre[1] - up), (centre[0], centre[1] + up),
                config.WHITE, 0.16 * alpha, 1)
        for sgn in (-1.0, 1.0):
            ov.dot((centre[0], centre[1] + up * sgn), 2.0, config.WHITE,
                   0.5 * alpha)

        r = radius * 1.42
        arm = r * 0.26
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cx = centre[0] + sx * r
            cy = centre[1] + sy * r
            ov.line((cx, cy), (cx - sx * arm, cy), config.WHITE, 0.34 * alpha, 1)
            ov.line((cx, cy), (cx, cy - sy * arm), config.WHITE, 0.34 * alpha, 1)

        if self._grab_hand is not None:
            ov.circle(centre, radius * 1.05, config.AMBER, 0.35 * alpha, 1)
        if self._pinch_ref is not None:
            ov.circle(centre, radius * 1.52, config.AMBER, 0.22 * alpha, 1)

    @staticmethod
    def _ellipse(ov, centre, rx, ry, color, alpha, steps=48):
        if alpha <= 0.006 or rx < 1.0:
            return
        th = np.linspace(0.0, 2.0 * np.pi, steps, dtype=np.float32)
        pts = np.stack([centre[0] + np.cos(th) * rx,
                        centre[1] + np.sin(th) * ry], axis=1)
        ov.polyline(pts, color, alpha, 1, closed=True)

    def _draw_probe(self, ov, centre, radius, alpha, t):
        p = self._probe
        d = np.asarray(centre, np.float32) - p
        dist = float(np.hypot(d[0], d[1]))
        if dist < 1e-3:
            return
        u = d / dist
        hit = np.asarray(centre, np.float32) - u * radius
        ov.line(p, hit, config.AMBER, 0.30 * alpha, 1)
        ov.circle(hit, 6.0 + 2.0 * math.sin(t * 8.0), config.AMBER,
                  0.75 * alpha, 1)
        ov.dot(hit, 1.8, config.WHITE, 0.9 * alpha)
        txt = "RANGE %4d" % int(max(0.0, dist - radius))
        ov.text((hit[0] + 12, hit[1] - 8), txt, config.AMBER, 0.6 * alpha,
                0.32, 1)

    # ------------------------------------------------------------------------
    def _draw_panel(self, ov, mesh, entry, alpha):
        if alpha < 0.06:
            return
        x, y = 28, 96
        lh = 15
        ov.text((x, y), "-- ORBIX", entry["accent"], 0.95 * alpha, 0.42, 1,
                shift=True)
        y += lh + 4
        rows = (
            ("MODEL", entry["label"]),
            ("TRIS", "%d / %d" % (self._drawn, mesh.n_tris)),
            ("VERTS", "%d" % len(mesh.verts)),
            ("SCALE", "%3d%%" % int(round(self.scale_shown * 100))),
            ("YAW", "%03d" % int(math.degrees(self.yaw) % 360)),
            ("PITCH", "%+03d" % int(math.degrees(self.pitch))),
            ("SPIN", "AUTO" if self.auto_spin else "HELD"),
            ("GPU", "SOFTWARE  %.1f ms" % self._ms),
        )
        for k, v in rows:
            ov.text((x, y), "%-6s %s" % (k, v), config.WHITE, 0.62 * alpha,
                    0.34, 1)
            y += lh

    def _draw_carousel(self, ov, alpha):
        n = len(self.library)
        if n < 2 or alpha < 0.06:
            return
        span = min(self.w * 0.52, 26.0 * n)
        x0 = (self.w - span) * 0.5
        y = self.h - 62
        ov.line((x0, y), (x0 + span, y), config.DIM, 0.20 * alpha, 1)
        for i in range(n):
            cx = x0 + span * (i + 0.5) / n
            hot = (i == self.index)
            col = self.library.entries[i]["accent"] if hot else config.DIM
            ov.line((cx, y - (7 if hot else 3)), (cx, y + (7 if hot else 3)),
                    col, (0.95 if hot else 0.30) * alpha, 2 if hot else 1)
            if hot:
                ov.circle((cx, y), 5.0, col, 0.7 * alpha, 1)
        name = self.label
        tw, _ = ov.measure(name, 0.42)
        ov.text(((self.w - tw) * 0.5, y + 26), name, config.WHITE,
                0.85 * alpha, 0.42, 1, shift=True)

    def _draw_error(self, ov, entry, alpha):
        lines = ["ORBIX MODEL UNAVAILABLE", entry["label"],
                 entry["error"][:58]]
        y = int(self.h * 0.45)
        for i, line in enumerate(lines):
            tw, _ = ov.measure(line, 0.42)
            ov.text(((self.w - tw) * 0.5, y + i * 22), line,
                    config.RED if i == 0 else config.DIM,
                    (0.9 if i == 0 else 0.6) * alpha, 0.42, 1)
        self._draw_carousel(ov, alpha)
