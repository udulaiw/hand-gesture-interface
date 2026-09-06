# Hand Interface

A full-screen, real-time computer-vision interface. The webcam feed *is* the
application: there is no window chrome, no panels and no dashboard. Your hands
are the controller.

```
webcam -> hand tracking -> smoothing -> gesture engine -> visual effects
```

Everything is custom-rendered. None of MediaPipe's default landmark drawing is
used.

Four instruments sit on top of that core, each on its own control axis:

| Instrument | Reached by | Changes |
| --- | --- | --- |
| Colour bar | pinching with **both** hands | the camera image |
| Accent pad | one hand in an **L** | the overlay colour |
| ORBIX viewer | an unlock **sequence** | a holographic `.glb` model you tumble |
| Boot sequence | launch | the `UDULAIW` title |

**[GUIDE.md](GUIDE.md) documents all four in full** — every gesture, every
tuning knob, how to add your own models, and how the software 3D renderer
works. What follows is the short version.

## Requirements

- Python 3.12
- A webcam
- `mediapipe >= 1.0`, `opencv-python`, `numpy`

## Setup

```bash
pip install -r requirements.txt
```

`mediapipe` 1.x ships no bundled model, so fetch the hand landmark bundle once
(~7.8 MB). This is the only step that needs an internet connection:

```bash
python setup_model.py
```

It downloads `hand_landmarker.task` from Google's MediaPipe model store into
`models/`, then pre-bakes every ORBIX `.glb` in `assets/orbix` into the
decimated `.npz` the holographic viewer loads, so no model stutters on its
first display. You can also place the landmark file there yourself. After this
the application runs entirely offline.

## Run

```bash
python main.py
```

Starts in a window. The interface is drawn in camera-frame coordinates and
scaled to whatever the window happens to be, so a small window is a first-class
way to run it, not a degraded one — drag it to any size and the hand positions
keep landing where the interface draws them. `F` toggles fullscreen, and
`--fullscreen` starts that way.

On launch the camera is *measured*, not just requested: see
[Camera negotiation](#camera-negotiation).

### Controls

| Key | Action |
| --- | --- |
| `ESC` / `Q` | exit |
| `H` | gesture legend |
| `D` | toggle debug overlay |
| `F` | toggle fullscreen / windowed |
| `O` | open / close the ORBIX viewer |
| `1`-`9` | jump to an ORBIX model |
| `N` / `B` | next / previous model |
| `S` | toggle the model's idle spin |
| `[` / `]` | previous / next accent |
| `G` | toggle palm grid |
| `P` | toggle particles |
| `T` | toggle fingertip trails |
| `C` | cycle the colour filter |
| `X` | clear the colour filter |
| `R` | reset all effects |

Any key during the boot sequence skips it.

### Command line

```bash
python main.py --fullscreen        # start fullscreen instead of windowed
python main.py --window-width 640   # smaller launch window (height follows)
python main.py --camera 1          # pick a different camera
python main.py --width 1920 --height 1080
python main.py --debug             # start with the debug overlay on
python main.py --selftest 8        # headless: check camera + throughput, then exit
python main.py --filter THERMAL    # start with a colour filter latched
python main.py --no-boot           # skip the UDULAIW title sequence
python main.py --theme 2           # start on the VIOLET accent
python main.py --model 3           # open the ORBIX viewer on SATURN
```

`--selftest` runs the real pipeline for N seconds with no window and reports
resolution, frame rate, inference latency, capture and inference rates,
detection rate and which gestures fired. Use it to diagnose a camera without a
display. Add `--capture out.png` to save the final composited frame.

### Camera negotiation

Webcam drivers accept resolutions they cannot actually deliver. Ask a typical
USB camera for 1280x720 and it will agree, then quietly fall back to
uncompressed YUY2 and send ten frames a second, because that is all the pixels
that fit down the cable at that rate. Nothing downstream can recover from it:
landmarks are only ever as fresh as the frames they came from, so 10 fps of
capture is 10 fps of tracking however fast the render loop spins — and it
presents as hands that keep vanishing and gestures that will not register, not
as a camera problem.

So `CameraStream` measures instead of trusting. Every candidate — Media
Foundation then DirectShow, MJPG then default, the requested size then the
`CAM_FALLBACKS` sizes — is opened and timed for `CAM_PROBE_SECONDS`, and the
first that sustains `CAM_MIN_FPS` wins. If none do, the fastest is kept. The
mode it settled on is printed at startup and shown in the debug overlay:

```
[camera] 1280x720  [msmf 1280x720 default 30 fps]
```

Only the properties a backend actually needs are sent, because each one is a
graph renegotiation that costs real time: Media Foundation reaches 720p30 from
the frame size alone, and adding FOURCC, FPS and BUFFERSIZE on top cost over a
second for an identical stream. The camera opens on its own thread while the
landmark model loads, since the two are independent and both slow. Startup
lands around five seconds, most of it the backend, and the boot sequence plays
after it.

## Gestures

| Gesture | Detected as | Response |
| --- | --- | --- |
| Open palm | four fingers extended | full hand visualisation, palm grid appears, particles orbit the palm |
| Point | index out, others folded | fingertip reticle plus a short trail |
| Pinch | thumb and index tips close | interaction ring with a closing progress arc, particle implosion and burst, confirmation flash |
| Fist | all fingers folded | skeleton collapses toward the palm, particles pull inward |
| Two fingers | index and middle out | draw mode — strokes persist and fade |
| Thumbs up | thumb out and pointing up, rest folded | confirmation flash |
| Wave | repeated side-to-side palm motion | clears drawn strokes |
| Double pinch | both hands pinching at once | the colour filter bar (below) |
| L frame | thumb and index splayed, rest folded | the accent pad opens in the wedge between them |
| Thumb + middle | | cycle the accent / hold the model's spin |
| Thumb + ring | | toggle particles / previous model |
| Thumb + pinky | | toggle trails / next model |
| Clench | any pose snapping shut into a fist | shockwave burst |
| Fist -> palm -> peace | one hand, inside 4 s | unlocks the ORBIX viewer |
| Both fists, held | | dismiss whatever is open |

Only one finger can be pinching at a time — see
[Pinch arbitration](#pinch-arbitration). The extra thumb pinches are
additionally gated on the index finger staying extended: a closing fist drags
all three tips past the thumb, and without that guard a fist would fire three
pinch events on the way down. The L is classified *before* POINT, because an L
satisfies POINT's conditions exactly and would otherwise be swallowed by it.

Press `H` in the app for the full table on screen, or see
**[GUIDE.md](GUIDE.md)**.

Detection is deliberately damped. A pose has to hold for a few frames before it
latches, and pinch uses separate enter/leave distances so it cannot chatter.

### Pinch arbitration

Testing each fingertip against its own threshold cannot work, and this is worth
being precise about because it is not a tuning problem. The middle fingertip
sits roughly one finger-width from the index fingertip, so a thumb-and-index
pinch puts the thumb *inside the middle finger's threshold as well* — both tests
pass, both fire, and the interface acts on a gesture nobody made. Tightening the
thresholds only trades that for pinches that refuse to register at all.

What separates the two is rank, not magnitude: in any real pinch exactly one
fingertip is nearest the thumb. So the contest runs first and the threshold
second.

1. The thumb-tip distance to all four fingertips is measured and normalised by a
   blend of hand size and that finger's own measured length, so a short pinky
   and a long middle finger read as "touching" at the same number.
2. The nearest tip wins, and it must beat the runner-up by `PINCH_WINNER_MARGIN`
   before it may take over. A thumb parked midway between two tips is ambiguous,
   and holds nothing — which is the correct answer.
3. Only the winner may hold a pinch. Everything downstream reads a single
   `pinch_finger` rather than four booleans that could disagree.
4. Releasing uses the wider threshold and ignores the margin, so a committed
   finger is not dropped because a neighbour drifted close; after a release,
   `PINCH_LOCKOUT` keeps the next finger from latching, so rolling the thumb
   across the fingertips does not fire a burst of gestures on the way past.

The index gets a one-frame hold because it is the gesture that must feel
instant; the others latch on two, since they open panels and switch models where
a frame is imperceptible and a false positive is not.

## The colour filter bar

Pinch thumb-to-index on **both hands at once** and a line snaps between the two
pinch points. That line is the control:

* **Length** scrubs the filter list. Hands together is `OFF`; spreading them
  wide walks through the ten entries. The active segment lights up and the bar
  is labelled as you go.
* **Angle** dials strength. A level bar is 50%; tilt the right hand up for more,
  down for less, out to a range of +/-38 degrees.

The preview is live, so the camera changes under your hands while you scrub.
Release either pinch and the pick latches — a small `FLT` chip in the
bottom-right corner shows what is running. `C` cycles the list from the
keyboard, `X` clears it.

| # | Filter | |
| --- | --- | --- |
| 1 | `OFF` | unfiltered |
| 2 | `CYAN DRIFT` | cool tint, red pulled down |
| 3 | `AMBER BURN` | warm tint, blue pulled down |
| 4 | `BLEACH` | crushed high-contrast wash |
| 5 | `INVERT` | negative |
| 6 | `INFRARED` | inferno colormap on luma |
| 7 | `NEON` | plasma colormap |
| 8 | `OCEAN` | ocean colormap |
| 9 | `TOXIC` | viridis colormap |
| 10 | `THERMAL` | turbo colormap |

The first five are pure colour matrices, so they are folded into the camera
grade at selection time and cost nothing per frame. The colormap five need
their own luma pass and cross-fade, which measures around 3 ms per frame at
720p; at full strength the cross-fade is skipped entirely.

Selection is smoothed and snaps with hysteresis. Raw hand distance crosses a
tick boundary several times a second even while you hold still, and without the
deadband the filter strobes between neighbours.

## Layout

```
main.py                  application loop, window, keys
config.py                palette and all tuning constants
setup_model.py           one-time model download
tracking/
  camera.py              threaded capture, always newest frame
  hand_tracker.py        MediaPipe Tasks HandLandmarker, async
  hand_state.py          derived per-hand geometry
  gestures.py            classification, hysteresis, events
  landmarks.py           landmark indices and the skeleton
effects/
  renderer.py            overlay canvas, bloom, camera grade + filters
  color_filter.py        two-hand pinch bar: filter selection and instrument
  hand_visualizer.py     bones, joint nodes, fingertips
  palm_grid.py           mesh mapped into the palm plane
  particles.py           vectorised particle field
  trails.py              fingertip wake and draw-mode strokes
  cursor.py              fingertip reticle, pinch ring
  scan.py                acquisition sweep
  hud.py                 status captions, debug panel, gesture legend
  theme.py               runtime accent palette, cross-faded
  lpad.py                the L-frame accent pad
  holo.py                ORBIX viewer + software wireframe renderer
  boot.py                the UDULAIW title sequence
utils/
  smoothing.py           One Euro filter, exponential averages
  fps.py                 frame timing
  glb.py                 dependency-free .glb reader with an .npz cache
assets/
  orbix/                 the shipped ORBIX .glb models
  cache/                 baked .npz, regenerated on demand
```

## How some of it works

**Smoothing.** Raw landmarks jitter at rest and lag under naive low-pass
filtering. Each hand runs a vectorised One Euro filter, which adapts its cutoff
to speed: heavy smoothing when the hand is still, light smoothing when it moves.
That is what makes the overlay feel welded to the hand. `FILTER_BETA` is the
whole game for feel — set it too low and a fast hand drags a rubber band behind
it, which reads as input lag even when the gesture logic fired on time.

**Hand identity.** Detections are matched to persistent hand slots by scoring
every (detection, slot) pair once and resolving the set cheapest-first, rather
than letting each detection grab its nearest free slot — which let one detection
claim the slot another wanted and swapped the two hands' identities mid-gesture.
Each slot's position is predicted forward by its own velocity before matching,
and the gate widens with speed, because a hand crossing the frame is not a new
hand. Handedness is a weighted vote rather than a per-frame assignment: MediaPipe
re-decides left versus right every frame and gets it wrong for a frame or two
whenever a hand rotates through edge-on, and one such frame used to be enough to
make the interface act on the wrong hand. A hand that goes missing keeps its
geometry for `HAND_LOST_SECONDS` before teardown, so a blink of lost tracking
does not drop a gesture that is still in progress.

**Frame gating.** The render loop runs two to three times faster than the camera
delivers frames, so inference is submitted only when the capture thread has
actually produced a new frame. Re-inferring a frame the tracker already has buys
nothing and takes the budget the next real frame needed. Between inferences the
hand states age their time-based values only — recomputing geometry from
unchanged landmarks produced identical numbers while feeding the velocity
estimator a zero every render frame, which quietly damped the measured speed
toward nothing.

**Palm grid.** The mesh is not a rectangle drawn around the hand. A quad is
built in the palm's own plane from the wrist and the index and pinky knuckles,
then each corner is pushed toward or away from the quad centre according to its
relative depth. That turns the parallelogram into a keystoned trapezoid, and a
homography from the unit square onto it carries every grid vertex along.
Rotation, scale and perspective therefore fall out of the geometry — turn your
hand and the mesh turns with it, move closer and it grows.

**Inference is decoupled from rendering.** The landmarker runs in LIVE_STREAM
mode with results arriving on a callback, so inference never blocks the frame
loop, and detection runs on a downscaled copy while everything draws at full
resolution. Only one frame is allowed in flight at a time; without that gate the
loop submits faster than inference drains and latency climbs steadily.

**The hologram is software.** No GPU, no 3D library. `utils/glb.py` parses
glTF-binary directly — seeking to the byte ranges the accessors name, so a 4 MB
file whose bulk is texture data never enters memory — and `effects/holo.py`
rotates, perspective-projects, back-face culls and depth-sorts the mesh in four
vectorised NumPy steps, then draws it in depth bands with batched OpenCV calls.
A wall-clock budget governs the triangle count, so a dense model costs detail
rather than frame rate. It draws onto the same overlay as everything else, so
the existing bloom pass is what makes it read as a hologram.

**Themes reach every effect** because effects read `config.CYAN` and
`config.AMBER` through the module at draw time rather than binding them at
import. Writing new values into those two names restyles the whole overlay on
the next frame.

**Rendering.** All effects draw into a single 8-bit overlay which is composited
onto the graded camera twice, once crisp and once blurred, so thin lines get
bloom without any per-effect glow work. The bloom is scaled while still
quarter-size, and the camera grade — desaturation, gain and lift — is one colour
matrix rather than a chain of passes.

## Performance

Measured at 1280x720 on the development machine:

| | |
| --- | --- |
| Display | ~56-58 fps |
| Full render stack | ~12.4 ms/frame |
| ...with the ORBIX hologram up | ~12.9 ms/frame |
| Hologram pass alone | ~2 ms/frame |
| Camera delivery | 30 fps |
| Inference latency | ~25-35 ms |

The number that matters most is the camera's: it bounds everything after it.
Before the negotiation described above, this machine was capturing at **10 fps**
while the render loop happily reported 85 — the interface looked fast and
tracked badly, because two out of every three gestures had no frame to happen
on. Render fps flatters the pipeline; the debug overlay (`D`) shows capture rate
and inference rate next to it for exactly that reason.

Effects also give way before frame rate does. A quality level tracks the frame
rate between `QUALITY_FPS_LOW` and `QUALITY_FPS_HIGH` — a wide band, so the
interface does not visibly pulse between detail levels — and sheds the particle
motion streak, the hologram's vertex dust and its wireframe antialiasing in that
order as it falls.

If you need more headroom on top of that, lower `HOLO_MS_BUDGET` (the hologram
loses detail, not smoothness), lower `DETECT_WIDTH` (tracking gets slightly less
reliable at distance), drop `PARTICLE_COUNT`, or raise `GLOW_DOWNSCALE`. If
tracking specifically is the problem, lower `CAM_WIDTH`/`CAM_HEIGHT` before
anything else: fewer pixels usually means a higher delivered frame rate, and
fresher landmarks beat sharper ones every time.

## Tuning

`config.py` holds everything worth changing: the palette, camera grade,
bloom, particle count, grid density, and the gesture thresholds including the
pinch enter/leave distances and how long a pose must hold.

## Troubleshooting

**"Hand landmark model not found"** — run `python setup_model.py`.

**"Could not open camera 0"** — another application is probably holding the
camera. Close it, or try `--camera 1`.

**Camera opens but the image is black** — some webcams need a moment to expose.
Check with `python main.py --selftest 5 --capture test.png`.

**Low frame rate** — see Performance above.

**`NO ORBIX MODELS FOUND`** — run `python setup_model.py`; it reports exactly
which models resolve. See [GUIDE.md](GUIDE.md) for the rest.
