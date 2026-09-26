# CS133 — Sphero BOLT: Pluto

A webcam-gesture-controlled Sphero BOLT ("Pluto") built for CS133. Uses
MediaPipe's Pose + Hand Landmarkers to recognize six gestures grouped into
two moods, reacts physically on the robot (LED matrix face, color, movement),
and plays a voice line for each — with escalating dialogue across repeated
interactions, a departure sequence after the 3rd, and real collision
detection with its own reaction line.

`pluto_final.py` is the current/complete version. The other `pluto_*.py`
files and `emotion_gestures.py` are earlier iterations kept for reference.

## Gestures

- **greet** — a raised-hand wave, a two-hand peace sign, or a thumbs up
- **antagonize** — a bear stance (arms up, elbows out, spread wide), a
  right-fist-into-left-palm strike, or a boxer's guard (both fists near the face)

Each category plays an opening line on its first two activations and a
farewell line from the 3rd activation onward; after the 3rd total
interaction, Pluto plays out a departure animation (calm amble if the last
mood was greet, an aggressive spin-then-flee if antagonize) and goes dark.

Bumping Pluto also triggers real BLE collision detection, with three
escalating "hurt" lines (calm → apologetic → fed up), rate-limited so one
scrape doesn't fire repeatedly.

## Setup

```
python -m venv my_env
my_env\Scripts\activate.bat
pip install spherov2 bleak opencv-contrib-python mediapipe
```

Download the two MediaPipe Tasks models into the project root:

```
curl -L -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

(Both are already committed to this repo, so this step is only needed if
you're starting fresh without them.)

### Audio

The `audio/` folder is **not** included in this repo (its voice lines are
ripped from a copyrighted game). To run any script that plays audio, create
an `audio/` folder in the project root and populate it with these exact
filenames — the scripts open them by name via Windows' MCI (`winmm.dll`), so
naming and format (mp3) must match:

```
hello-there.mp3
nice-day-huh.mp3
okay-guess-i-ll-see-you-later-then.mp3
who-let-this-idiot-run-free.mp3
-most-folk-wouldn-t-guess-you-were-a-moron.mp3
alright-i-m-going.mp3
i-feel-fine.mp3
ah-i-m-sorry-i-m-a-bit-distracted.mp3
stop-rijding-like-an-idiot-get-a-hold-of-yourself.mp3
```

## Running

```
python pluto_final.py
```

Shake the BOLT awake first. Once connected and the camera window opens:

- `w`/`a`/`s`/`d` — drive
- `x` — recalibrate "forward" to Pluto's current facing
- `r` — reset gesture/interaction counters
- `Esc` — quit

## Notes on the Windows/BLE environment

- Requires Python 3.14 compatibility shims for `spherov2` (an internal
  `functools.partialmethod` attribute it checks was renamed upstream).
- The BLE scan (`scanner.find_BOLT()`) must run on a thread that has never
  imported `mediapipe` or touched a camera — either poisons the main
  thread's COM state to STA, and `bleak`'s WinRT backend needs MTA. Each
  script here works around this by running the scan on a dedicated thread
  before anything else touches COM.
