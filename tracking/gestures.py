"""Gesture engine.

Classification is deliberately conservative: a candidate pose must hold for
several frames before it latches, so the interface never flickers between
states while a hand passes through an ambiguous shape. Discrete events
(pinch_start, wave, ...) are queued for the effects layer to react to.
"""
from collections import deque

import config
from tracking import landmarks as L

OPEN_PALM = "open_palm"
POINT = "point"
PINCH = "pinch"
FIST = "fist"
PEACE = "peace"
THUMBS_UP = "thumbs_up"
L_SHAPE = "l_shape"
NONE = "none"

LABELS = {
    OPEN_PALM: "OPEN PALM",
    POINT: "POINT",
    PINCH: "PINCH",
    FIST: "FIST",
    PEACE: "TWO FINGER",
    THUMBS_UP: "CONFIRM",
    L_SHAPE: "L FRAME",
    NONE: "-",
}

# Names carried by the ("extra_pinch", i, name) event, indexed the same way as
# HandState.EXTRA_FINGERS.
EXTRA_FINGERS = ("middle", "ring", "pinky")


def _classify(hand):
    e = hand.extension
    thumb, index, middle, ring, pinky = (float(x) for x in e)
    folded = (middle + ring + pinky) / 3.0

    if hand.pinch and index > 0.35:
        return PINCH
    # Thumbs-up must be tested before FIST: the four fingers are folded in both
    # poses, so a fist-first ordering swallows it every time.
    if thumb > 0.65 and index < 0.30 and folded < 0.25:
        # Only a genuinely upward thumb counts.
        up = hand.pts[L.THUMB_TIP][1] < hand.pts[L.THUMB_MCP][1] - hand.size * 0.25
        if up:
            return THUMBS_UP
    if max(index, middle, ring, pinky) < 0.30:
        return FIST
    if index > 0.68 and middle > 0.68 and ring < 0.38 and pinky < 0.45:
        return PEACE
    # The L must be tested before POINT: an L satisfies POINT's conditions
    # exactly (index out, other three folded) and would be swallowed by it. The
    # splay angle between the thumb and index is what separates the two.
    if hand.l_ok:
        return L_SHAPE
    if index > 0.70 and folded < 0.30:
        return POINT
    if (index + middle + ring + pinky) / 4.0 > 0.72:
        return OPEN_PALM
    return NONE


class _WaveDetector:
    """Counts direction reversals of horizontal palm motion."""

    def __init__(self):
        self.samples = deque()
        self.sign = 0
        self.cooldown = 0.0

    def update(self, hand, t, dt):
        self.cooldown = max(0.0, self.cooldown - dt)
        if not hand.visible or hand.size <= 0:
            self.samples.clear()
            return False

        vx = float(hand.velocity[0]) / hand.size
        sign = 1 if vx > config.WAVE_MIN_SPEED else (-1 if vx < -config.WAVE_MIN_SPEED else 0)
        if sign != 0 and sign != self.sign:
            self.sign = sign
            self.samples.append(t)

        while self.samples and t - self.samples[0] > config.WAVE_WINDOW:
            self.samples.popleft()

        if len(self.samples) >= config.WAVE_REVERSALS and self.cooldown <= 0.0:
            self.samples.clear()
            self.cooldown = 1.4
            return True
        return False


class _SequenceDetector:
    """Watches the stream of latched poses for an ordered unlock combination.

    Only poses that appear in the sequence are recorded, so incidental shapes
    the hand passes through between steps are ignored; anything that *is* in
    the sequence but arrives out of order restarts the attempt (or begins a new
    one, if it happens to be the first step). The whole thing must complete
    inside UNLOCK_WINDOW, which is what stops a fist made a minute ago from
    still counting toward an unlock now.
    """

    def __init__(self, sequence, window):
        self.sequence = tuple(sequence)
        self.window = float(window)
        self.step = 0
        self.started = 0.0

    def reset(self):
        self.step = 0

    @property
    def progress(self):
        return self.step / max(len(self.sequence), 1)

    def feed(self, pose, t):
        if pose not in self.sequence:
            return False
        if self.step and t - self.started > self.window:
            self.step = 0
        if pose == self.sequence[self.step]:
            if self.step == 0:
                self.started = t
            self.step += 1
            if self.step >= len(self.sequence):
                self.step = 0
                return True
        else:
            # A wrong-but-known pose is a failed attempt, except when it is
            # itself a fresh first step.
            self.step = 1 if pose == self.sequence[0] else 0
            if self.step:
                self.started = t
        return False


class GestureEngine:
    """Turns per-frame hand geometry into latched poses and discrete events.

    Events are ``(kind, hand_index, payload)`` tuples:

        pinch_start / pinch_end   index pinch, payload is the pinch point
        extra_pinch               thumb met middle / ring / pinky
        extra_release             ...and let go again
        gesture                   a pose latched, payload is the pose name
        clench                    the hand snapped shut into a fist
        wave                      three horizontal reversals in a second
        unlock                    the ORBIX pose sequence completed
        both_fists                both hands held closed past CLOSE_HOLD
    """

    def __init__(self, n_hands):
        self._candidate = [NONE] * n_hands
        self._count = [0] * n_hands
        self._prev_pinch = [False] * n_hands
        self._prev_extra = [[False, False, False] for _ in range(n_hands)]
        self._wave = [_WaveDetector() for _ in range(n_hands)]
        # One detector per hand, not one shared. A single shared detector is
        # fed by both hands at once, so the second hand's poses land in the
        # middle of the first hand's attempt and reset it — with two hands in
        # frame, which is the normal case, the sequence becomes almost
        # impossible to complete. The unlock is a one-handed gesture.
        self._sequence = [_SequenceDetector(config.UNLOCK_SEQUENCE,
                                            config.UNLOCK_WINDOW)
                          for _ in range(n_hands)]
        self._both_fists = 0.0
        self._both_fired = False
        self.events = []

    @property
    def unlock_progress(self):
        """0..1 through the unlock sequence on whichever hand is furthest."""
        return max((s.progress for s in self._sequence), default=0.0)

    def update(self, hands, t, dt):
        self.events.clear()
        for i, hand in enumerate(hands):
            if not hand.visible:
                self._count[i] = 0
                if hand.lost_time > config.HAND_LOST_SECONDS:
                    hand.gesture = NONE
                    self._candidate[i] = NONE
                if self._prev_pinch[i]:
                    self._prev_pinch[i] = False
                self._prev_extra[i] = [False, False, False]
                continue

            pose = _classify(hand)
            if pose == self._candidate[i]:
                self._count[i] += 1
            else:
                self._candidate[i] = pose
                self._count[i] = 1

            # PINCH latches instantly — it already has distance hysteresis and
            # feels broken if it lags behind the fingers. The L frame is held
            # longer than the rest, because it opens a panel and a false
            # positive is far more disruptive than a late one.
            if pose == PINCH:
                hold = 1
            elif pose == L_SHAPE:
                hold = config.L_HOLD_FRAMES
            else:
                hold = config.GESTURE_HOLD_FRAMES
            if self._count[i] >= hold and hand.gesture != pose:
                was = hand.gesture
                hand.gesture = pose
                hand.gesture_age = 0.0
                self.events.append(("gesture", i, pose))
                if pose == FIST and was not in (FIST, NONE):
                    # A snap shut from a shape that was already being held,
                    # rather than a hand that just entered frame closed.
                    self.events.append(("clench", i, tuple(hand.palm_center)))
                if self._sequence[i].feed(pose, t):
                    self.events.append(("unlock", i, None))
            else:
                hand.gesture_age += dt

            if hand.pinch and not self._prev_pinch[i]:
                self.events.append(("pinch_start", i, tuple(hand.pinch_point)))
            elif not hand.pinch and self._prev_pinch[i]:
                self.events.append(("pinch_end", i, tuple(hand.pinch_point)))
            self._prev_pinch[i] = hand.pinch

            for k, name in enumerate(EXTRA_FINGERS):
                now = bool(hand.extra_pinch[k])
                if now != self._prev_extra[i][k]:
                    kind = "extra_pinch" if now else "extra_release"
                    self.events.append((kind, i, name))
                    self._prev_extra[i][k] = now

            if self._wave[i].update(hand, t, dt):
                self.events.append(("wave", i, None))

        self._check_both_fists(hands, dt)
        return self.events

    def _check_both_fists(self, hands, dt):
        """Two closed fists, held. The universal 'dismiss' gesture."""
        closed = [h for h in hands if h.visible and h.gesture == FIST]
        if len(closed) >= 2:
            self._both_fists += dt
            if self._both_fists >= config.CLOSE_HOLD and not self._both_fired:
                self._both_fired = True
                self.events.append(("both_fists", 0, None))
        else:
            self._both_fists = 0.0
            self._both_fired = False

    @property
    def both_fists_progress(self):
        """0..1 toward the dismiss gesture firing, for the HUD ring."""
        if self._both_fired:
            return 1.0
        return min(1.0, self._both_fists / max(config.CLOSE_HOLD, 1e-3))
