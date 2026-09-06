"""Per-hand derived geometry.

Everything the visual layer needs is computed once, here, in pixel space and in
scale-invariant units. Effects never touch raw landmarks.
"""
import math

import numpy as np

import config
from tracking import landmarks as L
from utils.smoothing import Ema, OneEuro, clamp, smoothstep

# Ratio of |tip - mcp| to the finger's summed bone length. A straight finger
# approaches 1.0; a folded one collapses toward 0.45. Scale- and rotation-free.
_EXT_LO, _EXT_HI = 0.62, 0.92
_THUMB_LO, _THUMB_HI = 0.72, 0.95


class HandState:
    """A tracked hand that persists across frames."""

    __slots__ = (
        "label", "score", "pts", "z", "raw_pts", "visible", "lost", "age",
        "_filter", "_zfilter", "palm_center", "wrist", "size", "rotation",
        "tilt", "velocity", "speed", "extension", "openness", "pinch_dist",
        "pinch", "bbox", "presence", "_prev_center",
        "_presence_ema", "_size_ema", "scan_t", "gesture", "gesture_age",
        "index_tip", "thumb_tip", "pinch_point", "_rot_prev", "spin",
        # -- ORBIX extension -------------------------------------------------
        "pinch_dists", "extra_pinch", "l_angle", "l_vertex",
        "l_ok", "middle_tip", "pinky_tip",
        # -- pinch arbitration -------------------------------------------------
        "_seg", "_reach", "_tip_dist", "pinch_finger", "pinch_conf",
        "_pinch_cand", "_pinch_cand_frames", "_pinch_lock", "lost_time",
    )

    # Nominal length of each finger as a fraction of `size` (wrist to middle
    # knuckle), used to turn a measured finger into a fair reach. Only the
    # ratios matter; they exist so a pinky is not judged by an index's yardstick.
    _NOMINAL = (0.95, 0.78, 0.85, 0.80, 0.63)

    # Fingertip landmarks in arbitration order: index first, so a tie at equal
    # distance resolves to the index rather than to whatever numpy sorts first.
    _PINCH_TIPS = ("index", "middle", "ring", "pinky")

    # Thumb-to-fingertip contacts beyond the index, in the order they are
    # stored in `pinch_dists` / `extra_pinch`.
    EXTRA_FINGERS = ("middle", "ring", "pinky")

    def __init__(self, label):
        self.label = label
        self.score = 0.0
        self.pts = np.zeros((21, 2), np.float32)
        self.raw_pts = np.zeros((21, 2), np.float32)
        self.z = np.zeros(21, np.float32)
        self.visible = False
        self.lost = 999
        self.age = 0
        self._filter = OneEuro(config.FILTER_MIN_CUTOFF, config.FILTER_BETA,
                               config.FILTER_D_CUTOFF)
        self._zfilter = OneEuro(1.0, 0.02, 1.0)
        self.palm_center = np.zeros(2, np.float32)
        self.wrist = np.zeros(2, np.float32)
        self.index_tip = np.zeros(2, np.float32)
        self.thumb_tip = np.zeros(2, np.float32)
        self.pinch_point = np.zeros(2, np.float32)
        self.size = 1.0
        self.rotation = 0.0
        self.spin = 0.0
        self._rot_prev = None
        self.tilt = (0.0, 0.0)
        self.velocity = np.zeros(2, np.float32)
        self.speed = 0.0
        self.extension = np.zeros(5, np.float32)
        self.openness = 0.0
        self.pinch_dist = 1.0
        self.pinch = False
        self.bbox = (0, 0, 0, 0)
        self.presence = 0.0
        self._prev_center = None
        self._presence_ema = Ema(0.10, 0.0)
        self._size_ema = Ema(0.16)
        self.scan_t = 0.0
        self.gesture = "none"
        self.gesture_age = 0.0

        # -- ORBIX extension --------------------------------------------------
        self.middle_tip = np.zeros(2, np.float32)
        self.pinky_tip = np.zeros(2, np.float32)
        # Thumb tip to middle / ring / pinky tip, in hand sizes.
        self.pinch_dists = np.ones(3, np.float32)
        self.extra_pinch = [False, False, False]
        # The L pose: angle between the thumb and index axes, and the corner
        # they meet at. `l_ok` is the raw per-frame test; the gesture engine
        # adds the hold that turns it into a latched pose.
        self.l_angle = 0.0
        self.l_vertex = np.zeros(2, np.float32)
        self.l_ok = False

        # -- pinch arbitration -------------------------------------------------
        # Per-finger bone lengths and the reach they imply, refreshed each frame
        # so a hand turning edge-on is judged by what is currently visible.
        self._seg = np.ones(5, np.float32)
        self._reach = np.ones(5, np.float32)
        # Normalised thumb-tip distance to index / middle / ring / pinky tip.
        self._tip_dist = np.ones(4, np.float32)
        # At most one finger holds a pinch at a time. This is the whole answer
        # to "index pinch fires middle": they are no longer independent tests.
        self.pinch_finger = None
        self.pinch_conf = 0.0
        self._pinch_cand = None
        self._pinch_cand_frames = 0
        self._pinch_lock = 0.0
        self.lost_time = 0.0

    # -------------------------------------------------------------- update --
    def update(self, lm_xy, lm_z, score, width, height, t, dt):
        """Feed one frame of normalised landmarks."""
        raw = np.empty((21, 2), np.float32)
        raw[:, 0] = lm_xy[:, 0] * width
        raw[:, 1] = lm_xy[:, 1] * height
        self.raw_pts = raw
        self.pts = self._filter(raw, t).copy()
        # z is normalised roughly like x, so scale it the same way
        self.z = self._zfilter(lm_z * width, t).copy()
        self.score = score
        self.visible = True
        self.lost = 0
        self.lost_time = 0.0
        self.age += 1
        self._derive(dt)

    def mark_missing(self, dt):
        """A frame in which no detection claimed this slot.

        The teardown is on a wall-clock grace rather than a frame count. A hand
        crossing in front of the other, or turned edge-on for two frames, is a
        constant event; tearing the state down on it drops the pinch that is
        mid-gesture and replays the acquisition sweep. Holding the geometry for
        a fifth of a second means a blink costs nothing and a hand that has
        genuinely left still clears promptly.
        """
        self.lost += 1
        self.lost_time += dt
        self.visible = False
        if self.lost_time > config.HAND_LOST_SECONDS:
            self.age = 0
            self._filter.reset()
            self._zfilter.reset()
            self._prev_center = None
            self.pinch = False
            self.pinch_finger = None
            self._pinch_cand = None
            self._pinch_cand_frames = 0
            self._pinch_lock = 0.0
            self.gesture = "none"
            self.extra_pinch = [False, False, False]
            self.l_ok = False
        if self.lost > config.SCAN_RETRIGGER_FRAMES:
            self.scan_t = 0.0
        self.presence = self._presence_ema(0.0, dt)

    def tick(self, dt):
        """Age a held hand on a frame that carried no new inference.

        Only the time-based quantities move. Re-running the geometry over
        landmarks that have not changed produced no new numbers but did feed the
        velocity estimator a zero every render frame, quietly damping the
        measured speed toward nothing between inferences -- which then made the
        wake effect and the tracker's own motion prediction understate how fast
        the hand was actually moving.
        """
        self.presence = self._presence_ema(1.0, dt)
        self.gesture_age += dt
        self._pinch_lock = max(0.0, self._pinch_lock - dt)
        if self.scan_t < 1.0:
            self.scan_t = min(1.0, self.scan_t + dt / config.SCAN_DURATION)


    # ------------------------------------------------------------- geometry --
    def _derive(self, dt):
        p = self.pts
        self.wrist = p[L.WRIST]
        self.index_tip = p[L.INDEX_TIP]
        self.thumb_tip = p[L.THUMB_TIP]

        # Core scale: wrist to middle knuckle. Stable under finger motion.
        raw_size = float(np.linalg.norm(p[L.MIDDLE_MCP] - p[L.WRIST])) + 1e-3
        self.size = float(self._size_ema(raw_size, dt))
        s = self.size

        self.palm_center = p[list(L.PALM_LOOP)].mean(axis=0)

        v = p[L.MIDDLE_MCP] - p[L.WRIST]
        rot = math.atan2(float(v[1]), float(v[0]))
        if self._rot_prev is not None:
            d = (rot - self._rot_prev + math.pi) % (2 * math.pi) - math.pi
            self.spin = d / max(dt, 1e-3)
        self._rot_prev = rot
        self.rotation = rot

        # Palm tilt from relative depth, in hand-size units. Drives the grid's
        # perspective so the mesh leans with the palm.
        across = float(self.z[L.INDEX_MCP] - self.z[L.PINKY_MCP]) / s
        along = float(self.z[L.WRIST] - self.z[L.MIDDLE_MCP]) / s
        self.tilt = (clamp(across, -1.2, 1.2), clamp(along, -1.2, 1.2))

        # Finger extension. The summed bone length falls out of the same loop
        # and is kept, because pinch arbitration needs each finger's reach.
        ext = np.empty(5, np.float32)
        for i, chain in enumerate(L.FINGERS):
            seg = float(sum(
                np.linalg.norm(p[b] - p[a]) for a, b in zip(chain[:-1], chain[1:])
            )) + 1e-3
            span = float(np.linalg.norm(p[chain[-1]] - p[chain[0]]))
            r = span / seg
            lo, hi = (_THUMB_LO, _THUMB_HI) if i == 0 else (_EXT_LO, _EXT_HI)
            ext[i] = smoothstep(lo, hi, r)
            self._seg[i] = seg
        self.extension = ext
        self.openness = float(ext[1:].mean() * 0.82 + ext[0] * 0.18)

        self._derive_pinch(p, s, dt)
        self._derive_extras(p, s)

        # Motion
        c = self.palm_center
        if self._prev_center is not None and dt > 1e-4:
            inst = (c - self._prev_center) / dt
            self.velocity += (inst - self.velocity) * 0.35
        self._prev_center = c.copy()
        self.speed = float(np.linalg.norm(self.velocity)) / s   # hand-widths/s

        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        self.bbox = (float(x0), float(y0), float(x1), float(y1))

        self.presence = self._presence_ema(1.0, dt)
        if self.scan_t < 1.0:
            self.scan_t = min(1.0, self.scan_t + dt / config.SCAN_DURATION)

    # ---------------------------------------------------------------- pinch --
    def _derive_pinch(self, p, s, dt):
        """Decide which single finger, if any, is pinching the thumb.

        The old design tested each fingertip against its own threshold
        independently, which cannot work: the middle fingertip sits roughly one
        finger-width from the index fingertip, so a thumb-and-index pinch puts
        the thumb inside the middle finger's threshold as well, and both fire.
        Tightening the thresholds only trades that for pinches that refuse to
        register at all.

        What actually separates the two is rank, not magnitude. Exactly one
        fingertip is nearest the thumb in any real pinch, so the contest is run
        first and the threshold second: the nearest tip wins, it must beat the
        runner-up by PINCH_WINNER_MARGIN before it may take over, and only the
        winner is allowed to hold a pinch. Everything downstream reads a single
        `pinch_finger` instead of four booleans that could disagree.

        Distances are normalised by a blend of hand size and the finger's own
        measured length, so a short pinky and a long middle finger both read as
        "touching" at the same number.
        """
        self._pinch_lock = max(0.0, self._pinch_lock - dt)

        mix = float(config.PINCH_LENGTH_MIX)
        thumb = p[L.THUMB_TIP]
        # reach = what "one finger away" means for this finger on this hand.
        for i in range(5):
            nominal = self._seg[i] / max(self._NOMINAL[i], 1e-3)
            self._reach[i] = max((1.0 - mix) * s + mix * nominal, 1e-3)

        tips = (L.INDEX_TIP, L.MIDDLE_TIP, L.RING_TIP, L.PINKY_TIP)
        for k, tip in enumerate(tips):
            d = float(np.linalg.norm(p[tip] - thumb))
            self._tip_dist[k] = d / self._reach[k + 1]

        d = self._tip_dist
        order = np.argsort(d, kind="stable")
        win = int(order[0])
        runner = float(d[order[1]])
        best = float(d[win])
        name = self._PINCH_TIPS[win]
        # Clear separation, expressed as a ratio so it is scale-free. A pinch
        # held between two adjacent fingertips fails this and holds nothing,
        # which is the correct answer for an ambiguous pose.
        separated = best <= runner * config.PINCH_WINNER_MARGIN
        self.pinch_conf = clamp(1.0 - best / max(runner, 1e-3))

        held = self.pinch_finger
        if held is not None:
            # Releasing uses the wider threshold and ignores the margin: a
            # finger that has already committed should not be dropped because a
            # neighbour drifted close.
            hi = (config.PINCH_OFF if held == "index"
                  else config.PINCH_THRESHOLDS[held][1])
            k = self._PINCH_TIPS.index(held)
            if float(d[k]) < hi:
                self._pinch_cand = None
                self._pinch_cand_frames = 0
                self._apply_pinch(held, p)
                return
            self.pinch_finger = None
            self._pinch_lock = float(config.PINCH_LOCKOUT)

        ok = separated and self._pinch_lock <= 0.0
        if ok and name == "index":
            ok = best < config.PINCH_ON
            hold = int(config.PINCH_HOLD_INDEX)
        elif ok:
            on, _off = config.PINCH_THRESHOLDS[name]
            # A closing fist drags every fingertip past the thumb at once. The
            # index staying out is what tells a deliberate contact from a fist.
            ok = (best < on
                  and float(self.extension[1]) > config.PINCH_GUARD_INDEX)
            hold = int(config.PINCH_HOLD_EXTRA)
        else:
            hold = int(config.PINCH_HOLD_EXTRA)

        if not ok:
            self._pinch_cand = None
            self._pinch_cand_frames = 0
            self._apply_pinch(None, p)
            return

        if self._pinch_cand == name:
            self._pinch_cand_frames += 1
        else:
            self._pinch_cand = name
            self._pinch_cand_frames = 1
        if self._pinch_cand_frames >= hold:
            self._apply_pinch(name, p)
        else:
            self._apply_pinch(None, p)

    def _apply_pinch(self, name, p):
        """Publish the arbitration result into the fields effects read."""
        self.pinch_finger = name
        self.pinch = name == "index"
        for i, finger in enumerate(self.EXTRA_FINGERS):
            self.extra_pinch[i] = (name == finger)
        self.pinch_dist = float(self._tip_dist[0])
        self.pinch_dists[:] = self._tip_dist[1:]
        tip = {"index": L.INDEX_TIP, "middle": L.MIDDLE_TIP,
               "ring": L.RING_TIP, "pinky": L.PINKY_TIP}.get(name, L.INDEX_TIP)
        self.pinch_point = (p[L.THUMB_TIP] + p[tip]) * 0.5

    # ------------------------------------------------- ORBIX extension geometry
    def _derive_extras(self, p, s):
        """Thumb-to-middle/ring/pinky contacts, and the L-frame corner.

        Which finger is pinching is settled in `_derive_pinch`, which arbitrates
        all four contacts together; this only measures the L pose. Splitting the
        two apart is what stopped a thumb-and-index pinch from also registering
        as a thumb-and-middle one.
        """
        self.middle_tip = p[L.MIDDLE_TIP]
        self.pinky_tip = p[L.PINKY_TIP]

        # The L: thumb axis against index axis, measured from the joints they
        # actually pivot on rather than from the wrist, so a tilted forearm
        # does not change the reading.
        vt = p[L.THUMB_TIP] - p[L.THUMB_CMC]
        vi = p[L.INDEX_TIP] - p[L.INDEX_MCP]
        nt = float(np.linalg.norm(vt))
        ni = float(np.linalg.norm(vi))
        if nt > 1e-3 and ni > 1e-3:
            cos = float(np.dot(vt, vi)) / (nt * ni)
            self.l_angle = math.degrees(math.acos(clamp(cos, -1.0, 1.0)))
        else:
            self.l_angle = 0.0
        self.l_vertex = (p[L.THUMB_CMC] + p[L.INDEX_MCP]) * 0.5

        e = self.extension
        self.l_ok = bool(
            self.l_angle >= config.L_ANGLE_MIN
            and float(e[0]) >= config.L_THUMB_MIN
            and float(e[1]) >= config.L_INDEX_MIN
            and max(float(e[2]), float(e[3]), float(e[4])) <= config.L_FOLD_MAX
        )

    def l_basis(self):
        """(corner, thumb unit vector, index unit vector) of the L pose.

        The pad is drawn in the wedge these two vectors span, so it needs them
        normalised and in a fixed order regardless of which hand is held up.
        """
        p = self.pts
        corner = self.l_vertex
        vt = p[L.THUMB_TIP] - corner
        vi = p[L.INDEX_TIP] - corner
        nt = float(np.linalg.norm(vt)) or 1.0
        ni = float(np.linalg.norm(vi)) or 1.0
        return corner, (vt / nt).astype(np.float32), (vi / ni).astype(np.float32)

    # ------------------------------------------------------------- helpers --
    @property
    def active(self):
        return self.presence > 0.01

    def palm_basis(self):
        """(centre, across-vector, along-vector) spanning the palm in pixels."""
        p = self.pts
        mid = (p[L.INDEX_MCP] + p[L.PINKY_MCP]) * 0.5
        along = mid - p[L.WRIST]
        across = p[L.INDEX_MCP] - p[L.PINKY_MCP]
        return p[L.WRIST].copy(), across, along
