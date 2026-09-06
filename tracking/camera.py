"""Threaded webcam capture.

Reading straight from VideoCapture in the render loop makes latency creep up:
the driver buffers frames and you end up displaying the past. A grabber thread
that always keeps only the newest frame keeps the feed honest.
"""
import threading
import time

import cv2

import config


class CameraError(RuntimeError):
    pass


class CameraStream:
    def __init__(self, index=config.CAM_INDEX, width=config.CAM_WIDTH,
                 height=config.CAM_HEIGHT, fps=config.CAM_FPS):
        self.index = index
        self._cap, frame, self.mode = self._negotiate(index, width, height, fps)
        self.height, self.width = frame.shape[:2]
        self._frame = frame
        self._seq = 0
        self._lock = threading.Lock()
        self._running = True
        self._failures = 0
        self._grabbed = 0
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------- negotiation --
    @classmethod
    def _negotiate(cls, index, width, height, fps):
        """Open the camera at the best mode it will genuinely sustain.

        Drivers lie. Asking a webcam for 1280x720 gets an acknowledgement and a
        stream that arrives at ten frames a second, because the backend quietly
        settled for uncompressed YUY2, which cannot carry that many pixels over
        USB any faster. Nothing downstream can recover from it: landmarks are
        only ever as fresh as the frames they came from, so 10 fps of capture is
        10 fps of tracking however fast the render loop spins -- and it presents
        as hands that keep vanishing and gestures that will not register, not as
        a camera problem.

        So every candidate mode is measured rather than trusted. The first one
        that holds CAM_MIN_FPS wins; if none do, the fastest is kept.
        """
        candidates = []
        for backend, name in cls._backends():
            candidates.append((backend, name, width, height, config.CAM_FOURCC))
            candidates.append((backend, name, width, height, None))
        for fw, fh in config.CAM_FALLBACKS:
            if (fw, fh) == (width, height):
                continue
            for backend, name in cls._backends():
                candidates.append((backend, name, fw, fh, config.CAM_FOURCC))
        # Media Foundation reaches the requested mode from the size alone, and
        # each further property is a graph renegotiation costing well over a
        # second, so its FOURCC variant is pure startup cost for an identical
        # stream. Drop it and probe MSMF once.
        seen = set()
        pruned = []
        for backend, name, cw, ch, fourcc in candidates:
            key = (name, cw, ch, None if name == "msmf" else fourcc)
            if key in seen:
                continue
            seen.add(key)
            pruned.append((backend, name, cw, ch,
                           None if name == "msmf" else fourcc))
        candidates = pruned

        best = None            # (measured fps, cap, frame, mode)
        deadline = time.perf_counter() + config.CAM_NEGOTIATE_BUDGET
        for backend, name, cw, ch, fourcc in candidates:
            if best is not None and time.perf_counter() > deadline:
                break
            cap = cls._try_open(index, backend, cw, ch, fps, fourcc)
            if cap is None:
                continue
            rate, frame = cls._measure(cap)
            if frame is None:
                cap.release()
                continue
            mode = "%s %dx%d %s %.0f fps" % (name, frame.shape[1],
                                             frame.shape[0],
                                             fourcc or "default", rate)
            if rate >= config.CAM_MIN_FPS:
                if best is not None:
                    best[1].release()
                return cap, frame, mode
            if best is None or rate > best[0]:
                if best is not None:
                    best[1].release()
                best = (rate, cap, frame, mode)
            else:
                cap.release()

        if best is None:
            raise CameraError(
                "Could not open camera %d. Check that a webcam is connected "
                "and that no other application is using it." % index
            )
        return best[1], best[2], best[3]

    @staticmethod
    def _backends():
        # MSMF first: on Windows it is the backend that will negotiate a
        # compressed 720p stream, where DirectShow silently settles for 10 fps
        # of YUY2. DirectShow is still tried, because it is the more reliable of
        # the two on older devices and some virtual cameras MSMF refuses.
        out = []
        for attr, name in (("CAP_MSMF", "msmf"), ("CAP_DSHOW", "dshow"),
                           ("CAP_ANY", "any")):
            backend = getattr(cv2, attr, None)
            if backend is not None:
                out.append((backend, name))
        return out

    @staticmethod
    def _try_open(index, backend, width, height, fps, fourcc):
        """Open one candidate. Every property set is charged for, so only the
        ones a given backend actually needs are sent: on Media Foundation the
        size alone reaches 720p30, and adding FOURCC, FPS and BUFFERSIZE on top
        costs over a second of renegotiation for a stream that is identical.
        """
        try:
            cap = cv2.VideoCapture(index, backend)
        except cv2.error:
            return None
        if not cap.isOpened():
            cap.release()
            return None
        if fourcc:
            # FOURCC before the resolution: asking for MJPG afterwards makes
            # some drivers renegotiate straight back to their default mode.
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            except (cv2.error, TypeError):
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fourcc is not None:
            cap.set(cv2.CAP_PROP_FPS, fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
        return cap

    @staticmethod
    def _measure(cap):
        """Real delivered frame rate, and the last frame read."""
        frame = None
        for _ in range(5):          # let exposure and the pipeline settle
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        if frame is None:
            return 0.0, None
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < config.CAM_PROBE_SECONDS:
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                n += 1
        dt = time.perf_counter() - t0
        return (n / dt if dt > 0 else 0.0), frame

    def _pump(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._failures += 1
                if self._failures > 90:      # ~3 s of solid failure
                    self._running = False
                continue
            self._failures = 0
            with self._lock:
                self._frame = frame
                self._seq += 1
            self._grabbed += 1

    @property
    def alive(self):
        return self._running

    @property
    def capture_fps(self):
        """Measured delivery rate of the device, not of the render loop."""
        dt = time.perf_counter() - self._t0
        return self._grabbed / dt if dt > 0.25 else 0.0

    def read(self):
        with self._lock:
            return self._frame

    def read_new(self, last_seq):
        """(frame, seq) if a frame arrived since `last_seq`, else (None, seq).

        VideoCapture delivers at the device's rate while the render loop runs
        several times faster, so without this gate most loop iterations hand
        the tracker a frame it has already seen. Those duplicate inferences buy
        nothing and crowd out the one that carries new information.
        """
        with self._lock:
            if self._seq == last_seq:
                return None, self._seq
            return self._frame, self._seq

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
