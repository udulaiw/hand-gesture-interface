"""Central tuning + palette for the hand interface.

Colours are stored BGR because every draw call goes through OpenCV.
The palette is deliberately narrow: near-white structure, one cool accent,
one warm accent for "active" states, and a red used only as a status dot.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ---------------------------------------------------------------- camera ----
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
# Ask the driver for MJPG. Most USB webcams will only deliver 720p above about
# 10 fps in MJPG; left on the default YUY2 they negotiate the resolution and
# then quietly starve the tracker of frames, which reads as "the hand keeps
# dropping out" rather than as a camera problem.
CAM_FOURCC = "MJPG"
# Frame rate the negotiation is trying to reach. A driver will happily accept a
# resolution it can only deliver a handful of frames per second at, and report
# success, so the opener measures the rate it actually gets and moves on to the
# next candidate if the device is starving the tracker.
CAM_MIN_FPS = 24.0
CAM_PROBE_SECONDS = 0.55
# Total wall-clock the negotiation may spend before settling for the best mode
# it has found so far. In a dim room auto-exposure lengthens the shutter and
# every mode comes in under target, so without a ceiling the search would walk
# the whole candidate list and add seconds to startup for no gain.
CAM_NEGOTIATE_BUDGET = 3.0
# Fallback capture sizes, tried in order when the requested one cannot hold
# CAM_MIN_FPS. 960x540 is the sweet spot on most webcams: enough pixels for
# landmarks at arm's length, small enough to be delivered at full rate.
CAM_FALLBACKS = ((960, 540), (848, 480), (640, 480))
# Landmark inference runs on a downscaled copy; landmarks are normalised so
# the result maps back to the full-resolution frame for free. MediaPipe's own
# detector input is 192 px square, so anything above ~400 px only costs resize
# time without giving the model more to work with.
DETECT_WIDTH = 416

# --------------------------------------------------------------- tracking ---
MAX_HANDS = 2
# Acquisition is the expensive half of two-hand tracking: the palm detector only
# re-runs when fewer hands are being tracked than requested, so a high detection
# threshold is exactly what makes a second hand take seconds to appear. These sit
# at MediaPipe's own defaults; the temporal gates below do the rejecting instead,
# where a false positive costs one frame rather than a missed hand.
MIN_DETECTION_CONFIDENCE = 0.45
MIN_PRESENCE_CONFIDENCE = 0.45
MIN_TRACKING_CONFIDENCE = 0.45

# One Euro filter: low mincutoff kills jitter when still, beta restores
# responsiveness when the hand moves fast. Beta is the whole game for feel --
# too low and a fast hand drags a rubber band behind it, which reads as input
# lag even though the gesture logic fired on time.
FILTER_MIN_CUTOFF = 1.5
FILTER_BETA = 0.11
FILTER_D_CUTOFF = 1.0

# Seconds a hand may be missing before its visual state is torn down. Held as a
# duration rather than a frame count, so the grace does not silently shrink when
# the render loop speeds up -- and so a blink of lost tracking does not drop a
# gesture that is still in progress.
HAND_LOST_SECONDS = 0.22
# ...and how long it must be gone before the acquisition sweep is allowed to
# play again. Tracking drops out for a frame or two constantly; replaying the
# scan on every blip is the fastest way to make the interface look busy.
SCAN_RETRIGGER_FRAMES = 45

# --------------------------------------------------------------- palette ----
WHITE = (238, 242, 246)
DIM = (150, 158, 166)
CYAN = (208, 227, 79)       # #4FE3D0
AMBER = (66, 198, 255)      # #FFC642
RED = (92, 75, 255)         # #FF4B5C
INK = (28, 26, 24)

# ------------------------------------------------------------- compositing --
CAMERA_DESATURATE = 0.55    # 0 = untouched colour, 1 = fully monochrome
CAMERA_LIFT = -14           # slight crush so overlays read cleanly
CAMERA_GAIN = 1.06
VIGNETTE_STRENGTH = 0.34

CORE_GAIN = 1.0             # crisp overlay lines
GLOW_GAIN = 0.62            # bloom added on top
GLOW_SIGMA = 3.4
GLOW_DOWNSCALE = 4

# --------------------------------------------------------------- effects ----
PARTICLE_COUNT = 520
PARTICLE_DRAG = 0.955
TRAIL_LENGTH = 26
GRID_COLS = 6
GRID_ROWS = 7
SCAN_DURATION = 1.15        # seconds

# --------------------------------------------------------------- gestures ---
PINCH_ON = 0.34             # fraction of hand size — enter pinch
PINCH_OFF = 0.46            # leave pinch (hysteresis gap)
# Pinch distances are additionally normalised by the pinching finger's own
# length, so a long index and a short pinky read the same at contact. This is
# the weight of that correction: 0 = pure hand-size normalisation, 1 = pure
# finger-length. A blend beats either alone because finger length is itself
# noisy when the finger is curled.
PINCH_LENGTH_MIX = 0.45
GESTURE_HOLD_FRAMES = 3     # frames a gesture must persist before it latches
WAVE_MIN_SPEED = 0.85       # hand-widths per second
WAVE_REVERSALS = 3
WAVE_WINDOW = 1.1           # seconds

# ----------------------------------------------------------- colour filter --
# The two-hand pinch bar. Both hands pinch, and the line drawn between the two
# pinch points becomes the control: its LENGTH scrubs the filter list, its
# ANGLE dials strength. Releasing either pinch latches the current pick.
#
# Gap is measured between the pinch points as a fraction of frame width.
FILTER_GAP_MIN = 0.13       # hands together -> first entry (OFF)
FILTER_GAP_MAX = 0.66       # hands wide apart -> last entry
# Half-angle of the tilt range, in degrees. Level bar = 50% strength, tilted
# fully one way = 0%, the other way = 100%.
FILTER_TILT_RANGE = 38.0
FILTER_SNAP_MARGIN = 0.62   # how far past a tick before the pick changes
FILTER_SMOOTH = 0.09        # half-life, seconds, on the scrub + tilt readings

# Each entry is (name, kind, param, swatch).
#   kind "none"    unfiltered
#   kind "matrix"  3x4 BGR colour matrix, folded into the camera grade for free
#   kind "map"     OpenCV colormap applied to luma, cross-faded by strength
# Swatch is the BGR colour the entry's segment of the bar is drawn in.
FILTERS = (
    ("OFF",        "none",   None,                    (150, 158, 166)),
    ("CYAN DRIFT", "matrix", "cyan",                  (208, 227,  79)),
    ("AMBER BURN", "matrix", "amber",                 ( 66, 198, 255)),
    ("BLEACH",     "matrix", "bleach",                (232, 238, 244)),
    ("INVERT",     "matrix", "invert",                (180,  90, 210)),
    ("INFRARED",   "map",    "COLORMAP_INFERNO",      ( 70, 100, 250)),
    ("NEON",       "map",    "COLORMAP_PLASMA",       (220,  70, 230)),
    ("OCEAN",      "map",    "COLORMAP_OCEAN",        (230, 170,  50)),
    ("TOXIC",      "map",    "COLORMAP_VIRIDIS",      (110, 220, 130)),
    ("THERMAL",    "map",    "COLORMAP_TURBO",        ( 60, 190, 255)),
)

# Colour matrices for the "matrix" filters, in BGR. Rows are output channels,
# the trailing column is a constant offset. These are the *filter* pass; the
# camera grade already ran, so they compose on top of it.
FILTER_MATRICES = {
    "cyan":   ((1.18, 0.10, 0.00,  -4.0),
               (0.04, 1.06, 0.02,   2.0),
               (0.00, 0.06, 0.74, -10.0)),
    "amber":  ((0.70, 0.06, 0.00, -12.0),
               (0.02, 1.00, 0.10,   0.0),
               (0.00, 0.12, 1.24,   6.0)),
    "bleach": ((1.32, 0.06, 0.06, -34.0),
               (0.06, 1.32, 0.06, -34.0),
               (0.06, 0.06, 1.34, -32.0)),
    "invert": ((-1.0, 0.00, 0.00, 255.0),
               (0.00, -1.0, 0.00, 255.0),
               (0.00, 0.00, -1.0, 255.0)),
}


# ============================================================================
#  ORBIX EXTENSION
#  Everything below drives the additions layered on top of the original
#  interface: the accent-theme system, the L-frame colour pad, the extended
#  gesture vocabulary, the holographic model viewer, and the boot sequence.
# ============================================================================

# -------------------------------------------------------------- accent theme
# CYAN and AMBER above are the interface's two accent slots, and effects read
# them through the module at draw time rather than binding them at import. That
# makes the palette swappable at runtime: writing new values here restyles every
# overlay at once. THEMES is the list the L-pad scrubs through.
#
# Each entry is (name, accent, hot, dim_accent). `accent` replaces CYAN,
# `hot` replaces AMBER. All BGR.
THEMES = (
    ("AURORA",  (208, 227,  79), ( 66, 198, 255)),   # the original look
    ("EMBER",   ( 74, 142, 255), ( 92,  75, 255)),
    ("VIOLET",  (238, 120, 176), (255, 196, 120)),
    ("TOXIC",   (120, 236, 150), ( 70, 214, 236)),
    ("SIGNAL",  (236, 118, 236), (150, 240, 255)),
    ("ICE",     (255, 214, 150), (245, 245, 245)),
    ("SOLAR",   ( 60, 190, 255), ( 96, 240, 255)),
    ("ROSE",    (150, 120, 255), (210, 190, 255)),
)
THEME_FADE = 0.22            # half-life, seconds, of the accent cross-fade

# ------------------------------------------------------------- extra pinches
# Thumb-to-fingertip contacts other than the index. Distances are normalised by
# hand size exactly like PINCH_ON/PINCH_OFF, but each finger gets its own pair
# because a pinky reaches the thumb far more easily than a ring finger does.
#           (on,   off)
PINCH_THRESHOLDS = {
    "middle": (0.30, 0.42),
    "ring":   (0.33, 0.45),
    "pinky":  (0.36, 0.50),
}
# A closed fist puts every fingertip near the thumb, so an extra pinch only
# counts while the index finger is clearly still extended. This is the single
# guard that keeps the fist from firing three pinches on its way closed.
PINCH_GUARD_INDEX = 0.52

# ---------------------------------------------------------- pinch exclusivity
# The middle fingertip sits about a centimetre from the index fingertip, so a
# thumb-and-index pinch puts the thumb well inside the middle finger's raw
# threshold too. Thresholds alone cannot separate them however tight you set
# them; what separates them is that one contact is always clearly nearer than
# the other. Only the nearest fingertip may hold a pinch, and it must be nearer
# than the runner-up by this ratio before it is allowed to take over.
PINCH_WINNER_MARGIN = 0.78
# Frames of agreement before a non-index pinch latches. The index gets a
# shorter hold because it is the one gesture that must feel instant; the others
# open panels and switch models, where one extra frame is imperceptible and a
# false positive is not.
PINCH_HOLD_INDEX = 1
PINCH_HOLD_EXTRA = 2
# Once any pinch is live, a different finger cannot steal it until the live one
# has released. Without this, rolling the thumb across the fingertips fires a
# burst of gestures on the way past.
PINCH_LOCKOUT = 0.16         # seconds after a release before another may latch

# ------------------------------------------------------------------ L frame
# The "L": thumb and index extended and splayed, the other three folded. The
# colour pad opens in the wedge between them.
L_ANGLE_MIN = 46.0           # degrees between the thumb and index axes
L_FOLD_MAX = 0.38            # middle/ring/pinky extension ceiling
L_INDEX_MIN = 0.62           # index extension floor
L_THUMB_MIN = 0.52           # thumb extension floor
L_HOLD_FRAMES = 4            # frames the pose must survive before the pad opens

LPAD_R0 = 0.72               # inner radius, in hand sizes
LPAD_R1 = 1.20               # outer radius
# The fan is centred on the bisector of the thumb/index wedge, but its sweep is
# widened to at least this many degrees. The literal wedge of an L is only
# 50-70 degrees, and eight swatches inside that are slivers too thin to aim a
# fingertip at; opening the fan up keeps it growing out of the same corner
# while giving every swatch a target you can actually hit.
LPAD_SPAN_MIN = 96.0
LPAD_SPAN_MAX = 150.0
LPAD_OPEN = 0.16             # half-life, seconds, of the pad's open/close fade
# Twisting the wrist scrubs the swatch ring. Degrees of wrist roll that move
# the highlight by exactly one swatch.
LPAD_DIAL_STEP = 15.0
LPAD_DWELL = 0.45            # seconds the other hand must hover to commit

# ------------------------------------------------------------- unlock gesture
# The ORBIX viewer is deliberately behind a sequence rather than a single pose,
# so it can never open by accident while the hands are doing something else.
UNLOCK_SEQUENCE = ("fist", "open_palm", "peace")
UNLOCK_WINDOW = 4.0          # seconds allowed for the whole sequence
CLOSE_HOLD = 0.55            # seconds of two fists that dismisses the viewer

# ------------------------------------------------------------------- ORBIX --
ORBIX_DIR = os.path.join(ROOT, "assets", "orbix")
ORBIX_CACHE = os.path.join(ROOT, "assets", "cache")
# (file stem, display name, accent BGR). The models are shipped in
# assets/orbix; adding a .glb here is all it takes to put it in the carousel.
ORBIX_MODELS = (
    ("ORBIX_Planet",      "ORBIX PRIME", (208, 227,  79)),
    ("Jupiter",           "JUPITER",     ( 86, 176, 246)),
    ("Saturn",            "SATURN",      (120, 214, 244)),
    ("Nepthune",          "NEPTUNE",     (240, 168,  96)),
    ("Murcury",           "MERCURY",     (176, 186, 198)),
    ("Black_Hole",        "SINGULARITY", (226, 140, 210)),
    ("Cybertron",         "CYBERTRON",   (150, 236, 130)),
    ("Eura",              "EURA",        (238, 196, 110)),
    ("Simple_Rocket.001", "VOYAGER",     (120, 200, 255)),
)
ORBIX_MAX_TRIS = 6000        # decimation ceiling applied at load time
# Per-frame drawing budget, in triangles. The viewer culls back faces, sorts
# what is left by depth and draws only the nearest this many, so a dense model
# degrades in detail instead of in frame rate.
HOLO_TRI_BUDGET = 1500
HOLO_TRI_FLOOR = 420         # never thin out past this, however slow the frame
HOLO_MS_BUDGET = 4.5         # milliseconds the wireframe pass may take
HOLO_BANDS = 5               # depth bands, each drawn at its own alpha
HOLO_RADIUS = 0.20           # base radius as a fraction of the short frame edge
HOLO_FOV = 3.1               # eye distance in model radii; smaller = wider lens
HOLO_SPIN = 0.42             # idle yaw, radians per second
HOLO_DRAG = 2.6              # grab-rotate gain
HOLO_SCALE_MIN = 0.45
HOLO_SCALE_MAX = 2.30
HOLO_OPEN = 0.30             # half-life, seconds, of the open/close animation
HOLO_DUST = 0.55             # brightness of the vertex point cloud
HOLO_FOLLOW = 0.30           # how far the model drifts toward the hands, 0..1
# The model chases the hands through a critically damped spring rather than a
# fixed exponential. Stiffness sets how quickly it closes a large gap; the
# damping keeps it from overshooting, so a fast hand sweep is followed without
# the model wobbling to a stop afterwards.
HOLO_FOLLOW_STIFF = 26.0
HOLO_SCALE_SMOOTH = 0.07     # half-life, seconds, on the two-hand pinch scale
HOLO_SWAP_TIME = 0.34        # seconds of cross-dissolve when the model changes
# Models neighbouring the current one are parsed on a background thread while
# the viewer is idle, so stepping the carousel never pays a cold parse on the
# frame the gesture lands.
HOLO_PREFETCH = True

# ------------------------------------------------------------ adaptive detail
# Effects give way before frame rate does. Below the low mark the expensive
# extras (particle streaks, hologram dust, wireframe antialiasing) are shed one
# at a time; above the high mark they come back. The gap between the two is
# what stops the interface oscillating between quality levels.
QUALITY_FPS_LOW = 34.0
QUALITY_FPS_HIGH = 48.0

# --------------------------------------------------------------------- boot --
BOOT_TEXT = "UDULAIW"
BOOT_SUB = "ORBIX  //  HAND INTERFACE"
BOOT_DURATION = 4.2          # seconds, end to end
BOOT_LOG = (
    "boot: cold start",
    "optics: acquiring sensor",
    "mediapipe: hand landmarker  21 pt",
    "orbix: mounting model library",
    "gesture: vocabulary loaded",
    "overlay: bloom compositor ready",
    "link: established",
)
