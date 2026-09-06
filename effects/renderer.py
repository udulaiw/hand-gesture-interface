"""Overlay canvas and final compositing.

Every effect draws into one 8-bit canvas which is composited onto the graded
camera twice: once crisp, once blurred. That gives thin lines a soft bloom with
no per-effect glow work.

Two deliberate performance choices carry the frame budget at 720p:

  * The canvas is uint8, not float. A float canvas costs a full-resolution
    clip-and-convert every frame before it can be blended, which measured
    slower than everything else in the renderer combined.
  * The bloom is scaled while it is still quarter-size, so only the unavoidable
    upscale and the two blends touch full-resolution pixels.
"""
import cv2
import numpy as np

import config

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

# Luma weights in BGR order, used to build the desaturation matrix.
_LUMA = np.array([0.114, 0.587, 0.299], np.float32)


def _c(color, alpha):
    a = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
    return (color[0] * a, color[1] * a, color[2] * a)


def _pt(p):
    return (int(round(float(p[0]))), int(round(float(p[1]))))


class Overlay:
    """Drawing surface for all effects, cleared each frame."""

    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.canvas = np.zeros((height, width, 3), np.uint8)
        self._gw = max(1, width // config.GLOW_DOWNSCALE)
        self._gh = max(1, height // config.GLOW_DOWNSCALE)
        # Reused destinations: at 720p the per-frame allocation of these
        # buffers costs more than the arithmetic done on them.
        self._glow = np.empty((height, width, 3), np.uint8)
        self._small = np.empty((self._gh, self._gw, 3), np.uint8)
        # A second, coarser bloom buffer for when the frame budget is tight.
        # Halving the bloom's resolution again costs a slightly softer glow and
        # nothing else, which is a far better trade under load than dropping
        # frames -- and the interface is never without bloom, so the visual
        # language holds either way.
        self._cw = max(1, self._gw // 2)
        self._ch = max(1, self._gh // 2)
        self._coarse = np.empty((self._ch, self._cw, 3), np.uint8)

    def clear(self):
        self.canvas[:] = 0

    # ------------------------------------------------------------ primitives -
    def line(self, p0, p1, color, alpha=1.0, thickness=1):
        if alpha <= 0.004:
            return
        cv2.line(self.canvas, _pt(p0), _pt(p1), _c(color, alpha),
                 thickness, cv2.LINE_AA)

    def polyline(self, pts, color, alpha=1.0, thickness=1, closed=False):
        if alpha <= 0.004 or len(pts) < 2:
            return
        arr = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        if not np.isfinite(arr).all():
            return
        cv2.polylines(self.canvas, [np.int32(np.round(arr))], closed,
                      _c(color, alpha), thickness, cv2.LINE_AA)

    def circle(self, c, r, color, alpha=1.0, thickness=1):
        if alpha <= 0.004 or r < 0.4:
            return
        cv2.circle(self.canvas, _pt(c), int(round(r)), _c(color, alpha),
                   thickness, cv2.LINE_AA)

    def dot(self, c, r, color, alpha=1.0):
        self.circle(c, r, color, alpha, -1)

    def arc(self, c, r, a0, a1, color, alpha=1.0, thickness=1):
        if alpha <= 0.004 or r < 0.6:
            return
        cv2.ellipse(self.canvas, _pt(c), (int(r), int(r)), 0.0,
                    a0, a1, _c(color, alpha), thickness, cv2.LINE_AA)

    def fill_poly(self, pts, color, alpha=1.0):
        if alpha <= 0.004:
            return
        arr = np.asarray(pts, np.float32)
        if not np.isfinite(arr).all():
            return
        cv2.fillPoly(self.canvas, [np.int32(np.round(arr))],
                     _c(color, alpha), cv2.LINE_AA)

    def polys(self, batch, color, alpha=1.0, thickness=1, closed=True,
              antialias=True):
        """Draw many small polygons of equal vertex count in one call.

        `batch` is an (N, K, 2) array — N polygons of K points. This exists for
        the holographic wireframe, where N is in the thousands: one polylines
        call over the whole batch is several times cheaper than N line calls,
        and dropping antialiasing roughly halves it again. The bloom pass
        softens the resulting stair-stepping, so `antialias=False` is a real
        option rather than a visible downgrade.
        """
        if alpha <= 0.004 or batch is None or len(batch) == 0:
            return
        arr = np.asarray(batch)
        if arr.ndim != 3 or arr.shape[1] < 2:
            return
        if arr.dtype != np.int32:
            arr = np.round(arr).astype(np.int32)
        cv2.polylines(self.canvas, list(arr.reshape(len(arr), -1, 1, 2)),
                      closed, _c(color, alpha), thickness,
                      cv2.LINE_AA if antialias else cv2.LINE_8)

    def text(self, pos, s, color=config.WHITE, alpha=1.0, scale=0.42,
             thickness=1, mono=True, shift=False):
        """Small technical caption. `shift` adds a faint chromatic fringe."""
        if alpha <= 0.01 or not s:
            return
        font = FONT_MONO if mono else FONT
        fscale = scale * (1.55 if mono else 1.0)
        x, y = _pt(pos)
        if shift:
            cv2.putText(self.canvas, s, (x - 1, y), font, fscale,
                        _c(config.CYAN, alpha * 0.42), thickness, cv2.LINE_AA)
            cv2.putText(self.canvas, s, (x + 1, y), font, fscale,
                        _c(config.RED, alpha * 0.32), thickness, cv2.LINE_AA)
        cv2.putText(self.canvas, s, (x, y), font, fscale, _c(color, alpha),
                    thickness, cv2.LINE_AA)

    def measure(self, s, scale=0.42, thickness=1, mono=True):
        font = FONT_MONO if mono else FONT
        fscale = scale * (1.55 if mono else 1.0)
        (w, h), _ = cv2.getTextSize(s, font, fscale, thickness)
        return w, h

    def splat(self, xy, intensity, color):
        """Point scatter — the particle system's fast path."""
        if len(xy) == 0:
            return
        x = np.round(xy[:, 0]).astype(np.int32)
        y = np.round(xy[:, 1]).astype(np.int32)
        ok = ((x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
              & np.isfinite(intensity))
        if not ok.any():
            return
        x, y = x[ok], y[ok]
        i = np.clip(intensity[ok], 0.0, 1.0).astype(np.float32)
        col = np.asarray(color, np.float32)
        add = i[:, None] * col[None, :]
        cur = self.canvas[y, x].astype(np.float32)
        self.canvas[y, x] = np.clip(cur + add, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------ composite --
    def composite(self, base_bgr, quality=1.0):
        """Blend the overlay onto `base_bgr`, crisp plus bloom.

        `quality` below the halfway mark drops the bloom to a coarser buffer.
        This pass is the single largest item in the render budget -- the two
        full-resolution adds and the upscale are irreducible -- so it is also
        the one worth giving a cheaper mode.
        """
        core = self.canvas
        if quality < 0.5:
            buf, bw, bh = self._coarse, self._cw, self._ch
            sigma = config.GLOW_SIGMA * 0.5
        else:
            buf, bw, bh = self._small, self._gw, self._gh
            sigma = config.GLOW_SIGMA
        cv2.resize(core, (bw, bh), dst=buf, interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(buf, (0, 0), sigma, dst=buf)
        # Scale the bloom while it is still a fraction of the pixels.
        if abs(config.GLOW_GAIN - 1.0) > 1e-3:
            cv2.convertScaleAbs(buf, dst=buf, alpha=config.GLOW_GAIN)
        cv2.resize(buf, (self.w, self.h), dst=self._glow,
                   interpolation=cv2.INTER_LINEAR)

        # base_bgr is the graded frame, which the grader owns and reuses.
        cv2.add(base_bgr, self._glow, dst=base_bgr)
        if abs(config.CORE_GAIN - 1.0) > 1e-3:
            core = cv2.convertScaleAbs(core, alpha=config.CORE_GAIN)
        cv2.add(base_bgr, core, dst=base_bgr)
        return base_bgr


def _compose(after, before):
    """Compose two 3x4 colour transforms: apply `before`, then `after`."""
    a, ao = after[:, :3], after[:, 3:]
    b, bo = before[:, :3], before[:, 3:]
    return np.hstack([a @ b, a @ bo + ao]).astype(np.float32)


class CameraGrade:
    """Desaturate, crush and vignette the feed so overlays sit on top of it.

    Desaturation, gain and lift are one 3x4 colour matrix, so the whole grade is
    a single pass over the frame plus the vignette multiply.

    The user-selected colour filter rides on the same pass where it can. A
    matrix filter is composed into the grade matrix at selection time and costs
    nothing per frame; a colormap filter needs its own luma pass and cross-fade,
    which is why the two kinds are kept apart instead of unified behind a LUT.
    """

    def __init__(self, width, height):
        self._vignette = self._build(width, height)
        self._shape = (height, width)
        self._base = self._build_matrix()
        self._matrix = self._base
        self._out = np.empty((height, width, 3), np.uint8)
        self._colormap = None
        self._mix = 0.0
        self._gray = np.empty((height, width), np.uint8)
        self._mapped = np.empty((height, width, 3), np.uint8)
        self._filter = (0, 0.0)

    @staticmethod
    def _build_matrix():
        d = float(config.CAMERA_DESATURATE)
        m = (1.0 - d) * np.eye(3, dtype=np.float32) + d * np.tile(_LUMA, (3, 1))
        m *= float(config.CAMERA_GAIN)
        offset = np.full((3, 1), float(config.CAMERA_LIFT), np.float32)
        return np.hstack([m, offset]).astype(np.float32)

    # ----------------------------------------------------------- filtering --
    def set_filter(self, index, strength):
        """Select one of config.FILTERS at 0..1 strength.

        Cheap to call every frame: it early-outs unless the pick actually moved
        enough to matter, so the live preview during a drag can just push the
        current reading in without the caller tracking changes.
        """
        index = int(index) % len(config.FILTERS)
        strength = 0.0 if strength < 0.0 else (1.0 if strength > 1.0 else float(strength))
        if self._filter[0] == index and abs(self._filter[1] - strength) < 0.004:
            return
        self._filter = (index, strength)

        _, kind, param, _ = config.FILTERS[index]
        if kind == "none" or strength <= 0.002:
            self._matrix = self._base
            self._colormap = None
            self._mix = 0.0
            return

        if kind == "matrix":
            f = np.asarray(config.FILTER_MATRICES[param], np.float32)
            ident = np.hstack([np.eye(3, dtype=np.float32),
                               np.zeros((3, 1), np.float32)])
            blended = ident + (f - ident) * strength
            self._matrix = _compose(blended, self._base)
            self._colormap = None
            self._mix = 0.0
        else:
            self._matrix = self._base
            # An OpenCV build without a given colormap degrades to no filter
            # rather than crashing the interface mid-gesture.
            self._colormap = getattr(cv2, param, None)
            self._mix = strength if self._colormap is not None else 0.0

    @staticmethod
    def _build(w, h):
        ax = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        ay = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        gx, gy = np.meshgrid(ax, ay)
        r = np.sqrt(gx * gx + gy * gy) / 1.4142
        v = 1.0 - config.VIGNETTE_STRENGTH * np.clip(r, 0, 1) ** 2.1
        v = np.clip(v, 0.0, 1.0)
        return (v * 255.0).astype(np.uint8)[:, :, None].repeat(3, axis=2)

    def __call__(self, frame):
        if frame.shape[:2] != self._shape:
            self._vignette = self._build(frame.shape[1], frame.shape[0])
            self._shape = frame.shape[:2]
            self._out = np.empty_like(frame)
            self._gray = np.empty(self._shape, np.uint8)
            self._mapped = np.empty_like(frame)
        cv2.transform(frame, self._matrix, dst=self._out)
        if self._colormap is not None and self._mix > 0.002:
            cv2.cvtColor(self._out, cv2.COLOR_BGR2GRAY, dst=self._gray)
            if self._mix > 0.985:
                # Fully dialled up: map straight into the output and skip the
                # cross-fade, which is a third of this path's cost at 720p.
                cv2.applyColorMap(self._gray, self._colormap, dst=self._out)
            else:
                cv2.applyColorMap(self._gray, self._colormap, dst=self._mapped)
                cv2.addWeighted(self._out, 1.0 - self._mix, self._mapped,
                                self._mix, 0.0, dst=self._out)
        cv2.multiply(self._out, self._vignette, dst=self._out, scale=1.0 / 255.0)
        return self._out
