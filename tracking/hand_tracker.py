"""MediaPipe Tasks hand landmarker, wrapped for a real-time loop.

mediapipe >= 1.0 removed the legacy `mp.solutions` API, so this targets the
Tasks HandLandmarker. It runs in LIVE_STREAM mode: frames are pushed in and
results arrive on a callback, so inference never blocks the render loop.
Detected hands are matched to persistent HandState objects by proximity, which
survives the occasional handedness flip.
"""
import os
import threading
import time

import cv2
import numpy as np

import config
from tracking.hand_state import HandState


class ModelMissing(RuntimeError):
    pass


def ensure_model(path=config.MODEL_PATH):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    raise ModelMissing(
        f"Hand landmark model not found at {path}.\n"
        "Run:  python setup_model.py   (one-time, needs internet)"
    )


class HandTracker:
    def __init__(self, max_hands=config.MAX_HANDS):
        import mediapipe as mp

        self._mp = mp
        model = ensure_model()
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            result_callback=self._on_result,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._result = None
        self._result_id = 0
        self._consumed_id = 0
        self._last_ts = -1
        self.hands = [HandState(f"hand{i}") for i in range(max_hands)]
        self.latency_ms = 0.0
        self._sent = {}
        # Frames are submitted faster than inference drains them, so without a
        # gate the queue grows and every landmark arrives progressively later.
        # Holding one frame in flight keeps latency at the cost of one
        # inference, not a backlog.
        self._inflight = False
        self._inflight_since = 0.0
        # Measured rate of completed inferences, which is the number that
        # actually bounds gesture latency -- the render loop's own fps does not.
        self.infer_fps = 0.0
        self._last_result_at = 0.0
        self.dropped = 0
        # Sticky handedness. MediaPipe re-decides left vs right every frame and
        # gets it wrong for a frame or two whenever a hand rotates through
        # edge-on, so the raw label is treated as a vote rather than as truth.
        self._label_votes = [0] * max_hands

    # ------------------------------------------------------------ callback --
    def _on_result(self, result, image, timestamp_ms):
        sent_at = self._sent.pop(timestamp_ms, None)
        self._inflight = False
        now = time.perf_counter()
        with self._lock:
            self._result = result
            self._result_id += 1
            if sent_at is not None:
                ms = (now - sent_at) * 1000.0
                self.latency_ms += (ms - self.latency_ms) * 0.2
            if self._last_result_at:
                gap = now - self._last_result_at
                if gap > 1e-4:
                    self.infer_fps += (1.0 / gap - self.infer_fps) * 0.15
            self._last_result_at = now

    # --------------------------------------------------------------- submit --
    def submit(self, frame_bgr):
        """Push a frame for asynchronous inference, unless one is pending."""
        now = time.perf_counter()
        if self._inflight:
            # Safety valve: if a result never came back, resume submitting.
            if now - self._inflight_since < 0.5:
                self.dropped += 1
                return
            self._inflight = False
        h, w = frame_bgr.shape[:2]
        if w > config.DETECT_WIDTH:
            scale = config.DETECT_WIDTH / w
            small = cv2.resize(frame_bgr, (config.DETECT_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.perf_counter() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        self._sent[ts] = time.perf_counter()
        if len(self._sent) > 32:
            self._sent.clear()
        try:
            self._inflight = True
            self._inflight_since = now
            self._landmarker.detect_async(image, ts)
        except Exception:
            self._inflight = False
            # A dropped frame is survivable; never let it kill the loop.

    # -------------------------------------------------------------- consume --
    def poll(self, width, height, t, dt):
        """Fold the newest result into the persistent hand states."""
        with self._lock:
            result = self._result
            rid = self._result_id
        fresh = rid != self._consumed_id
        self._consumed_id = rid

        detections = []
        if result is not None and fresh:
            for i, lms in enumerate(result.hand_landmarks):
                xy = np.array([[p.x, p.y] for p in lms], np.float32)
                z = np.array([p.z for p in lms], np.float32)
                label, score = "?", 0.0
                if i < len(result.handedness) and result.handedness[i]:
                    cat = result.handedness[i][0]
                    label, score = cat.category_name, cat.score
                detections.append((xy, z, label, score))

        if fresh:
            self._assign(detections, width, height, t, dt)
        else:
            # No new inference this frame: age the time-based state only. The
            # geometry cannot have changed, so recomputing it would cost a full
            # derive per render frame for an identical answer.
            for hand in self.hands:
                if hand.visible:
                    hand.tick(dt)
        return self.hands

    def _assign(self, detections, width, height, t, dt):
        """Match this frame's detections to the persistent hand slots.

        Two things went wrong with a per-detection "nearest free slot" search.
        A detection could claim the slot another detection wanted and push it
        onto an idle one, which swapped the two hands' identities mid-gesture;
        and because handedness was copied through blindly, a single frame where
        MediaPipe called the left hand right was enough to make the interface
        act on the wrong hand.

        So: every (detection, slot) pair is scored once, the whole set is
        resolved cheapest-first, and handedness is a weighted vote rather than
        a per-frame assignment. With only two hands the sort is trivially small.
        """
        n = len(self.hands)
        # Predict where each held hand should be by now. A hand moving quickly
        # travels a long way between inferences; matching against its last known
        # position is what makes fast movement look like tracking loss.
        pred = []
        for hand in self.hands:
            if hand.age == 0:
                pred.append(None)
                continue
            lag = max(dt, 1e-3)
            pred.append(hand.wrist + hand.velocity * min(lag, 0.05))

        pairs = []
        for di, (xy, _z, label, score) in enumerate(detections):
            px = np.array([xy[0, 0] * width, xy[0, 1] * height], np.float32)
            for hi in range(n):
                if pred[hi] is None:
                    continue
                hand = self.hands[hi]
                d = float(np.linalg.norm(pred[hi] - px))
                # The gate widens with speed for the same reason the position is
                # predicted: a hand crossing the frame is not a new hand.
                gate = max(hand.size * 2.5, 120.0) + hand.speed * hand.size * 0.25
                if d > gate:
                    continue
                cost = d
                # Agreeing with the established handedness is worth about a
                # third of the gate, so it breaks near-ties between two hands
                # close together without overriding clear proximity.
                if hand.label in ("Left", "Right") and score > 0.6:
                    cost += (-0.30 if label == hand.label else 0.30) * gate
                pairs.append((cost, di, hi))

        pairs.sort(key=lambda item: item[0])
        taken_d, taken_h, matched = set(), set(), {}
        for _cost, di, hi in pairs:
            if di in taken_d or hi in taken_h:
                continue
            taken_d.add(di)
            taken_h.add(hi)
            matched[di] = hi

        # Detections with no home take an idle slot, preferring one whose last
        # known handedness agrees, so a hand that left and came back lands in
        # the slot it had before.
        for di in range(len(detections)):
            if di in matched:
                continue
            label = detections[di][2]
            free = [k for k in range(n)
                    if k not in taken_h and self.hands[k].age == 0]
            if not free:
                free = [k for k in range(n) if k not in taken_h]
            if not free:
                continue
            hi = next((k for k in free if self.hands[k].label == label), free[0])
            taken_h.add(hi)
            matched[di] = hi

        for di, hi in matched.items():
            xy, z, label, score = detections[di]
            hand = self.hands[hi]
            self._vote_label(hi, hand, label, score)
            hand.update(xy, z, score, width, height, t, dt)

        for hi, hand in enumerate(self.hands):
            if hi not in taken_h:
                hand.mark_missing(dt)

    def _vote_label(self, hi, hand, label, score):
        """Sticky left/right: a flip must be sustained before it is believed."""
        if label not in ("Left", "Right"):
            return
        if hand.age == 0 or hand.label not in ("Left", "Right"):
            hand.label = label
            self._label_votes[hi] = 2
            return
        if label == hand.label:
            self._label_votes[hi] = min(6, self._label_votes[hi] + 1)
            return
        # Confident disagreement counts double; a hesitant one barely counts.
        self._label_votes[hi] -= 2 if score > 0.85 else 1
        if self._label_votes[hi] <= 0:
            hand.label = label
            self._label_votes[hi] = 2

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
