# UDULAIW / ORBIX — full guide

The hand interface, extended. Four instruments now sit on top of the original
tracking core, each on its own control axis so no two of them ever fight over
the same gesture:

| Instrument | How you reach it | What it changes |
|---|---|---|
| **Colour bar** | pinch with **both** hands | the **camera** image |
| **Accent pad** | one hand in an **L** | the **overlay** colour |
| **ORBIX viewer** | an unlock **sequence** | a holographic `.glb` model you tumble |
| **Boot sequence** | on launch | the `UDULAIW` title |

Nothing was removed. The two-hand colour bar behaves exactly as it did.

---

## 1. Run it

```bash
cd C:\Users\Dell\Desktop\hand-interface
python setup_model.py
python main.py
```

`setup_model.py` is a one-time step. It fetches the MediaPipe landmark bundle
and pre-bakes every ORBIX `.glb` into the decimated `.npz` the viewer actually
loads, so no model ever stutters on its first display. `run_hand_interface.bat`
does all of this for you.

Useful flags:

```bash
python main.py --fullscreen        # take the whole screen (default is a window)
python main.py --window-width 640   # launch smaller; the height follows the camera
python main.py --no-boot           # skip the title sequence
python main.py --theme 2           # start on the VIOLET accent
python main.py --model 3           # open the viewer on SATURN
python main.py --filter THERMAL    # start with a camera filter latched
python main.py --debug             # numbers on
python main.py --selftest 6        # headless: camera, fps, latency, then exit
```

---

## 2. The boot sequence

Four overlapping stages across ~4.2 seconds:

1. a scan line crosses the black, leaving raster bars behind it
2. `UDULAIW` resolves **letter by letter** out of a glitching character
   cipher — each letter lands with a chromatic red/cyan split and throws a
   shock ring
3. `ORBIX // HAND INTERFACE` tracks out underneath while boot lines type
   themselves down the left margin, each ticking over to `OK`
4. corner brackets snap in, `SYSTEM ONLINE` flashes, and the camera bleeds up
   through the black as the veil lifts

**Any key skips it** — it jumps to the handover rather than cutting, so you
never go straight from black to a live camera.

Tune it in `config.py`:

```python
BOOT_TEXT     = "UDULAIW"                  # the title itself
BOOT_SUB      = "ORBIX  //  HAND INTERFACE"
BOOT_DURATION = 4.2                        # seconds, end to end
BOOT_LOG      = (...)                      # the log lines
```

---

## 3. The accent pad — the L-frame colour picker

**Make an L with one hand:** thumb out, index up, the other three folded. A
fan of eight colour swatches unfolds into the empty wedge between your thumb
and index finger, anchored to the corner where they meet — so it rides and
rotates with your hand.

Two ways to drive it, both live at the same time:

- **Dial (one-handed).** Twist your wrist. The roll accumulated since the pad
  opened scrubs the highlight around the fan. **Pinch** — close the L, thumb to
  index — to lock it in.
- **Point (two-handed).** Put your *other* hand's index fingertip on a swatch.
  A direct hit takes over from the dial immediately, and a short dwell (0.45 s,
  shown as a closing ring on your fingertip) commits without a second gesture.
  Pinching with the pointing hand commits instantly.

Browsing is free: the interface previews each accent live as you scrub, and if
you drop the pose without committing it snaps back to whatever was active
before you opened the pad.

The eight accents: `AURORA` (the original), `EMBER`, `VIOLET`, `TOXIC`,
`SIGNAL`, `ICE`, `SOLAR`, `ROSE`.

> **This is a different axis from the colour bar.** The pad recolours the
> *instrumentation* — skeleton, palm grid, particles, reticles, HUD, hologram
> frame. The two-hand bar grades the *camera image*. They compose, and neither
> disables the other.

The mechanism is worth knowing if you extend the code: every effect reads
`config.CYAN` and `config.AMBER` through the module **at draw time**, never
binding them at import. `effects/theme.py` writes new values into those two
names and the whole overlay restyles on the next frame, cross-faded over
0.22 s so it reads as a retune rather than a glitch.

---

## 4. The ORBIX holographic viewer

### Unlocking it

Perform this sequence **with one hand**, inside 4 seconds:

> **FIST → OPEN PALM → TWO FINGERS (peace)**

Three pips at the top of the frame fill in as you go, and the next pose you
need is named underneath them. Poses that aren't part of the sequence are
ignored, so you can pass through anything in between; a sequence pose in the
*wrong* order restarts the attempt.

It's behind a sequence, not a single pose, for a reason: the viewer rebinds
several gestures while it's open, and it must be impossible to trigger by
accident. Each hand runs its own detector, so your second hand can be doing
anything at all.

`O` on the keyboard does the same thing.

### Controlling it

| Gesture | Action |
|---|---|
| **fist + move** | tumble the model. Release and it keeps spinning — there's inertia on the throw |
| **pinch with both hands, spread/close** | scale it (45 %–230 %) |
| **thumb + pinky** | next model |
| **thumb + ring** | previous model |
| **thumb + middle** | toggle the idle spin |
| **point** | a range-finder line casts onto the hull with a live distance readout |
| **both fists, held ~0.5 s** | dismiss. A closing red ring shows the hold |

Keyboard: `1`–`9` jump straight to a model, `N`/`B` step, `S` toggles spin,
`O` closes.

While the viewer is open the two-hand pinch belongs to it, so the colour bar
stands down — silently, without latching anything, and it comes straight back
with the filter you already had once the viewer closes.

### The models

Nine of your ORBIX assets are shipped in `assets/orbix`:

| # | Model | Source file |
|---|---|---|
| 1 | ORBIX PRIME | `ORBIX_Planet.glb` |
| 2 | JUPITER | `Jupiter.glb` |
| 3 | SATURN | `Saturn.glb` |
| 4 | NEPTUNE | `Nepthune.glb` |
| 5 | MERCURY | `Murcury.glb` |
| 6 | SINGULARITY | `Black_Hole.glb` |
| 7 | CYBERTRON | `Cybertron.glb` |
| 8 | EURA | `Eura.glb` |
| 9 | VOYAGER | `Simple_Rocket.001.glb` |

**Adding more.** Drop a `.glb` into `assets/orbix` and add a line to
`ORBIX_MODELS` in `config.py`:

```python
ORBIX_MODELS = (
    ...
    ("Polyphemus", "POLYPHEMUS", (150, 200, 240)),   # file stem, label, BGR accent
)
```

Then run `python setup_model.py` again to bake its cache. That's the whole
procedure — no other file needs touching. A model whose file is missing is
skipped at startup; one that fails to parse shows its error in place of the
hologram instead of taking the app down.

The 56 MB `Black hole.glb` is deliberately not shipped — the 102 KB
`Black_Hole.glb` is the same object with a fraction of the texture payload, and
the viewer draws wireframe, so the textures are dead weight either way.

### How the rendering works

`utils/glb.py` is a dependency-free glTF-binary reader — no `trimesh`, no
`pygltflib`. It parses the node hierarchy, triangle topology, vertex positions
and material base colours, and skips textures, animation, skins and morph
targets. It reads the binary chunk by **seeking** to the byte ranges the
accessors name, so a 4 MB file whose bulk is PNG data never gets pulled into
memory. Every load is cached as an `.npz` keyed by the source file's size and
mtime, so the parse happens once ever.

`effects/holo.py` then draws it, per frame, in four vectorised NumPy steps and
a handful of batched OpenCV calls:

1. rotate every vertex by the current yaw / pitch / roll
2. project with a real perspective divide, so the near face of a planet
   genuinely swells toward the camera
3. cull back faces from the sign of each projected triangle's area — **which**
   sign is decided per frame by comparing the mean depth of the two halves,
   because these exports don't agree on winding order
4. depth-sort the survivors, keep the nearest N, and draw them in five bands,
   each at its own alpha, so the far side of the shell recedes

Step 4 is also the performance governor. `HOLO_MS_BUDGET` is a wall-clock
budget for the wireframe pass; overrun and N comes down, so a dense model or a
slow machine costs you **depth, not frame rate**. Measured here: ~3 ms per
frame for the whole hologram, against a ~15 ms full-stack frame.

Everything is drawn onto the same overlay canvas as the rest of the interface,
so it picks up the existing bloom compositor for free — which is what makes it
read as a hologram rather than as line art.

---

## 5. The complete gesture vocabulary

Press **`H`** in the app for this same table on screen.

### Core (unchanged)

| Gesture | Action |
|---|---|
| pinch, thumb + index | grab; gathers particles, confirmation flash |
| pinch, **both** hands | colour bar — length picks the filter, tilt sets strength |
| open palm | palm mesh, particles orbit at a stand-off ring |
| two fingers | draw mode — strokes persist for seconds |
| point | cursor reticle on the fingertip |
| thumbs up | confirm |
| wave (3 reversals in ~1 s) | clear strokes |
| fist | collapse — skeleton contracts, particles implode |

### New

| Gesture | Closed | ORBIX open |
|---|---|---|
| **L frame** (thumb + index splayed) | accent pad | accent pad |
| **thumb + middle** | cycle accent | toggle idle spin |
| **thumb + ring** | toggle particles | previous model |
| **thumb + pinky** | toggle trails | next model |
| **fist, snapped shut** | shockwave burst | shockwave burst |
| **fist + move** | — | tumble the model |
| **both fists, held** | closes the legend | dismiss the viewer |
| **fist → palm → peace** | unlocks the viewer | — |

**Why the extra pinches can't misfire.** A closing fist drags your middle, ring
and pinky tips right past your thumb, which would fire three pinch events on the
way down. Every extra pinch is therefore gated on the **index finger staying
extended** (`PINCH_GUARD_INDEX`), and each finger carries its own
on/off threshold pair — a pinky reaches the thumb far more easily than a ring
finger does. This is verified by test: a fist held for 30 frames produces zero
extra-pinch events.

**Why the L is tested before POINT.** An L satisfies POINT's conditions exactly
(index out, other three folded), so a point-first ordering swallows it every
time. The splay angle between the thumb and index axes — measured from the
joints they pivot on, not from the wrist, so a tilted forearm doesn't change the
reading — is what separates them. The L also holds for more frames than other
poses before latching, because it opens a panel and a false positive is more
disruptive than a late one.

---

## 6. Keyboard

| Key | Action |
|---|---|
| `ESC` / `Q` | quit |
| `H` | gesture legend |
| `D` | debug readout |
| `F` | fullscreen |
| `O` | open / close the ORBIX viewer |
| `1`–`9` | jump to a model |
| `N` / `B` | next / previous model |
| `S` | toggle the model's idle spin |
| `[` / `]` | previous / next accent |
| `C` | cycle the camera filter |
| `X` | camera filter off |
| `G` | palm grid |
| `P` | particles |
| `T` | trails |
| `R` | reset everything |

Any key during the boot sequence skips it and does nothing else.

---

## 7. Tuning

Every knob is in `config.py`, grouped under `ORBIX EXTENSION`.

**Gestures won't trigger:**

```python
PINCH_THRESHOLDS   # per-finger (on, off) for thumb+middle/ring/pinky
PINCH_GUARD_INDEX  # lower it if extra pinches feel unresponsive
L_ANGLE_MIN        # degrees of thumb/index splay needed for the L
L_HOLD_FRAMES      # frames the L must survive before the pad opens
UNLOCK_WINDOW      # seconds allowed for the whole unlock sequence
CLOSE_HOLD         # seconds of two fists that dismisses
```

**Accent pad feel:**

```python
LPAD_DIAL_STEP     # degrees of wrist twist per swatch — lower = twitchier
LPAD_DWELL         # seconds of hover to commit
LPAD_R0, LPAD_R1   # fan radii, in hand sizes
LPAD_SPAN_MIN/MAX  # how far the fan opens past the literal wedge
```

**Hologram:**

```python
HOLO_RADIUS        # size as a fraction of the short frame edge
HOLO_TRI_BUDGET    # triangles per frame (ceiling)
HOLO_MS_BUDGET     # the wall-clock budget the governor holds to
HOLO_FOV           # eye distance in radii — smaller is a wider lens
HOLO_SPIN, HOLO_DRAG, HOLO_FOLLOW
HOLO_BANDS, HOLO_DUST
```

If frame rate is the problem, `HOLO_MS_BUDGET` is the one to turn down — it
degrades detail rather than smoothness. `PARTICLE_COUNT`, `GLOW_DOWNSCALE` and
`DETECT_WIDTH` are the next biggest levers, and all three predate this work.

---

## 8. What was added, file by file

| File | Status | Purpose |
|---|---|---|
| `utils/glb.py` | new | dependency-free `.glb` reader + `.npz` cache |
| `effects/holo.py` | new | the holographic viewer and its software renderer |
| `effects/lpad.py` | new | the L-frame accent pad |
| `effects/theme.py` | new | runtime accent palette with cross-fade |
| `effects/boot.py` | new | the `UDULAIW` title sequence |
| `assets/orbix/` | new | the nine shipped models |
| `assets/cache/` | new | baked `.npz`, regenerated on demand |
| `tracking/hand_state.py` | extended | extra pinch distances, L-frame geometry |
| `tracking/gestures.py` | extended | `L_SHAPE`, extra pinches, clench, per-hand unlock sequence, two-fist dismiss |
| `effects/hud.py` | extended | legend, unlock pips, dismiss ring, accent chip |
| `effects/renderer.py` | extended | `Overlay.polys` — batched polygon draw |
| `effects/color_filter.py` | extended | `enabled=` so the viewer can borrow the two-hand pinch |
| `effects/particles.py` | fixed | resolves palette at draw time, so themes reach it |
| `config.py` | extended | the whole `ORBIX EXTENSION` block |
| `main.py` | rewired | instruments, event routing, new keys, boot veil |
| `setup_model.py` | extended | pre-bakes the ORBIX caches |

Dependencies are unchanged: `mediapipe`, `opencv-python`, `numpy`. Nothing was
added to load or draw the 3D models.

---

## 9. Troubleshooting

**`NO ORBIX MODELS FOUND`** — `assets/orbix` is empty or the names in
`ORBIX_MODELS` don't match the files. Run `python setup_model.py` to see
exactly which ones resolve.

**`ORBIX MODEL UNAVAILABLE` with an error line** — that one file failed to
parse. The rest of the carousel still works. Draco-compressed exports are the
usual cause; re-export without compression.

**The unlock never fires** — hold each pose long enough to *latch* (watch the
bottom-left status line change), and complete all three inside 4 seconds. Use
one hand. Press `O` to confirm the viewer itself works.

**The accent pad won't open** — fold your middle, ring and pinky properly and
splay the thumb wide; the pose needs a real right angle, not a loose point.
Raise `L_FOLD_MAX` or lower `L_ANGLE_MIN` if your hand sits outside the range.

**Extra pinches feel dead** — keep your index finger extended. That's the fist
guard doing its job. Raise the relevant entry in `PINCH_THRESHOLDS`.

**The hologram looks sparse** — that's the back-face cull plus the triangle
budget. Raise `HOLO_TRI_BUDGET`, or `HOLO_MS_BUDGET` if the governor is the one
throttling it (the panel's `GPU SOFTWARE n ms` row tells you which).

**Frame rate dropped** — check the `D` overlay. `infer` is MediaPipe and
unrelated to any of this; if the hologram's `GPU` row is at its budget, lower
`HOLO_MS_BUDGET`.
