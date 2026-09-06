"""A lightweight field of drifting points that reacts to the hands.

The whole pool is one set of NumPy arrays updated with vector maths, and drawn
by scattering into the overlay rather than by per-particle draw calls. That is
what keeps several hundred particles nearly free at 720p.

Behaviour per gesture:
  open palm  orbit the palm at a stand-off radius
  fist       collapse inward, heavily damped
  pinch      converge on the pinch point; releasing scatters them
  motion     particles inside the hand's wake get carried along
"""
import numpy as np

import config
from tracking.gestures import FIST, OPEN_PALM, PINCH

WHITE, CYAN, AMBER = 0, 1, 2


def _color(kind):
    """Resolved at draw time, not import time, so a theme change reaches here."""
    if kind == CYAN:
        return config.CYAN
    if kind == AMBER:
        return config.AMBER
    return config.WHITE


class ParticleField:
    def __init__(self, width, height, count=config.PARTICLE_COUNT):
        self.w = width
        self.h = height
        self.n = count
        rng = np.random.default_rng(7)
        self.rng = rng
        self.pos = np.stack([rng.uniform(0, width, count),
                             rng.uniform(0, height, count)], axis=1).astype(np.float32)
        self.vel = rng.normal(0, 6, (count, 2)).astype(np.float32)
        self.ttl = rng.uniform(2.0, 7.0, count).astype(np.float32)
        self.life = rng.uniform(0.0, 1.0, count).astype(np.float32) * self.ttl
        self.kind = np.zeros(count, np.uint8)
        self.bright = rng.uniform(0.25, 0.85, count).astype(np.float32)
        self.phase = rng.uniform(0, 6.283, count).astype(np.float32)
        self._acc = None

    # ------------------------------------------------------------------------
    def reset(self):
        self.pos[:, 0] = self.rng.uniform(0, self.w, self.n)
        self.pos[:, 1] = self.rng.uniform(0, self.h, self.n)
        self.vel[:] = self.rng.normal(0, 6, (self.n, 2))
        self.life[:] = self.rng.uniform(0.0, 1.0, self.n) * self.ttl
        self.kind[:] = WHITE

    def burst(self, point, count=70, speed=460.0, kind=AMBER):
        """Scatter the oldest particles outward from a point."""
        idx = np.argsort(self.life)[:count]
        ang = self.rng.uniform(0, 2 * np.pi, idx.size).astype(np.float32)
        mag = self.rng.uniform(0.35, 1.0, idx.size).astype(np.float32) * speed
        self.pos[idx] = np.asarray(point, np.float32)
        self.vel[idx, 0] = np.cos(ang) * mag
        self.vel[idx, 1] = np.sin(ang) * mag
        self.life[idx] = self.ttl[idx] * self.rng.uniform(0.55, 1.0, idx.size)
        self.kind[idx] = kind
        self.bright[idx] = self.rng.uniform(0.6, 1.0, idx.size)

    def implode(self, point, count=90, kind=CYAN):
        """Fling particles inward from a ring — the pinch 'gather'."""
        idx = np.argsort(self.life)[:count]
        ang = self.rng.uniform(0, 2 * np.pi, idx.size).astype(np.float32)
        rad = self.rng.uniform(90, 240, idx.size).astype(np.float32)
        p = np.asarray(point, np.float32)
        self.pos[idx, 0] = p[0] + np.cos(ang) * rad
        self.pos[idx, 1] = p[1] + np.sin(ang) * rad
        self.vel[idx, 0] = -np.cos(ang) * rad * 2.2
        self.vel[idx, 1] = -np.sin(ang) * rad * 2.2
        self.life[idx] = self.ttl[idx]
        self.kind[idx] = kind
        self.bright[idx] = self.rng.uniform(0.5, 1.0, idx.size)

    # ------------------------------------------------------------------------
    def update(self, hands, t, dt):
        if dt <= 0:
            return
        # One reused accumulator. At 520 particles the allocation is not large,
        # but it happened every frame for the life of the session and it is
        # free to avoid.
        if self._acc is None:
            self._acc = np.zeros_like(self.vel)
        acc = self._acc
        acc[:] = 0.0
        active = [h for h in hands if h.active]

        for hand in active:
            c = hand.palm_center.astype(np.float32)
            d = self.pos - c
            dist = np.sqrt((d * d).sum(axis=1)) + 1e-3
            reach = max(hand.size * 4.2, 140.0)
            near = dist < reach
            if not near.any():
                continue
            fall = np.clip(1.0 - dist / reach, 0.0, 1.0) ** 1.6
            nx = d[:, 0] / dist
            ny = d[:, 1] / dist
            w = fall * hand.presence

            g = hand.gesture
            if g == PINCH:
                target = hand.pinch_point.astype(np.float32)
                td = target - self.pos
                tdist = np.sqrt((td * td).sum(axis=1)) + 1e-3
                pull = w * 1500.0 / np.maximum(tdist, 26.0)
                acc[:, 0] += td[:, 0] / tdist * pull
                acc[:, 1] += td[:, 1] / tdist * pull
            elif g == FIST:
                acc[:, 0] -= nx * w * 900.0
                acc[:, 1] -= ny * w * 900.0
                self.vel[near] *= (1.0 - 0.55 * dt * 60.0 / 60.0)
            elif g == OPEN_PALM:
                # Tangential orbit plus a soft spring holding a stand-off ring.
                ring = hand.size * 1.9
                radial = (dist - ring) * -3.4
                acc[:, 0] += (-ny * 260.0 + nx * radial) * w
                acc[:, 1] += (nx * 260.0 + ny * radial) * w
            else:
                acc[:, 0] += nx * w * 60.0
                acc[:, 1] += ny * w * 60.0

            # Wake: fast hands drag nearby particles with them.
            sp = float(np.linalg.norm(hand.velocity))
            if sp > 90.0:
                drag = w * np.clip(sp / 900.0, 0.0, 1.4)
                acc[:, 0] += hand.velocity[0] * drag * 2.4
                acc[:, 1] += hand.velocity[1] * drag * 2.4

        self.vel += acc * dt
        damp = config.PARTICLE_DRAG ** (dt * 60.0)
        self.vel *= damp
        np.clip(self.vel, -1600.0, 1600.0, out=self.vel)
        self.pos += self.vel * dt

        self.life -= dt
        dead = self.life <= 0.0
        # Off-screen particles are recycled too, so the pool never idles away.
        dead |= ((self.pos[:, 0] < -60) | (self.pos[:, 0] > self.w + 60) |
                 (self.pos[:, 1] < -60) | (self.pos[:, 1] > self.h + 60))
        k = int(dead.sum())
        if k:
            self._respawn(np.flatnonzero(dead), active)

    def _respawn(self, idx, active):
        rng = self.rng
        k = idx.size
        self.life[idx] = self.ttl[idx] * rng.uniform(0.6, 1.0, k)
        self.bright[idx] = rng.uniform(0.2, 0.8, k).astype(np.float32)
        self.kind[idx] = WHITE
        if active:
            # Prefer to reseed in a ring around a hand so density follows it.
            hand = active[int(rng.integers(len(active)))]
            ang = rng.uniform(0, 2 * np.pi, k)
            rad = rng.uniform(hand.size * 1.2, hand.size * 4.0, k)
            self.pos[idx, 0] = hand.palm_center[0] + np.cos(ang) * rad
            self.pos[idx, 1] = hand.palm_center[1] + np.sin(ang) * rad
        else:
            self.pos[idx, 0] = rng.uniform(0, self.w, k)
            self.pos[idx, 1] = rng.uniform(0, self.h, k)
        self.vel[idx] = rng.normal(0, 8, (k, 2))

    # ------------------------------------------------------------------------
    def draw(self, ov, t, quality=1.0):
        """`quality` below 1 sheds the motion streak, which is half the cost."""
        fade = np.clip(self.life / np.maximum(self.ttl, 1e-3), 0.0, 1.0)
        # Ease in and out so nothing pops.
        fade = np.sin(np.clip(fade, 0, 1) * np.pi) ** 0.55
        twinkle = 0.75 + 0.25 * np.sin(t * 3.1 + self.phase)
        inten = (self.bright * fade * twinkle * 0.55).astype(np.float32)

        streak = quality > 0.35
        prev = self.pos - self.vel * 0.010 if streak else None
        for kind in (WHITE, CYAN, AMBER):
            m = self.kind == kind
            if not m.any():
                continue
            col = _color(kind)
            ov.splat(self.pos[m], inten[m], col)
            if streak:
                ov.splat(prev[m], inten[m] * 0.45, col)
