"""Webcam gesture recognizer using mediapipe's Pose + Hand Landmarkers (Tasks
API — the legacy mp.solutions.hands/pose API doesn't exist in mediapipe 1.0.1).

Four gestures:
  greet       — a raised hand waving side to side (a hello wave), OR
                both hands showing a peace sign (index+middle up)
  antagonize  — both arms raised, elbows out, hands spread wide
                (a bear's threat/defensive stance), OR
                a right-hand fist struck into an open left palm
"""
import os, time, random, ctypes, itertools
from collections import deque
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

# --- Arthur Morgan voice lines, MCI (winmm.dll) playback — no extra package
# needed, and it plays asynchronously so it never blocks the camera loop.
# Activations 1 and 2 of a gesture each use a different opener (shuffled once
# per sequence, so it's not always the same order) — no immediate repeats.
# Activation 3 onward = the farewell line for that category.
GREET_OPENERS = ["hello-there.mp3", "nice-day-huh.mp3"]
GREET_FAREWELL = "okay-guess-i-ll-see-you-later-then.mp3"
ANTAGONIZE_OPENERS = ["who-let-this-idiot-run-free.mp3", "-most-folk-wouldn-t-guess-you-were-a-moron.mp3"]
ANTAGONIZE_FAREWELL = "alright-i-m-going.mp3"

_mci = ctypes.windll.winmm
_alias_pool = itertools.cycle(range(4))   # small rotating pool so back-to-back clips don't collide

def play_mp3(path):
    """Opens, queries the real clip length, and plays it. Returns the length
    in seconds so callers can wait for it to actually finish instead of
    guessing a fixed duration."""
    alias = f"voice{next(_alias_pool)}"
    _mci.mciSendStringW(f'close {alias}', None, 0, None)   # harmless if nothing was open
    _mci.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
    buf = ctypes.create_unicode_buffer(64)
    _mci.mciSendStringW(f'status {alias} length', buf, 64, None)
    try:
        length_sec = int(buf.value) / 1000
    except ValueError:
        length_sec = 3.0   # fallback if the length query ever fails
    _mci.mciSendStringW(f'play {alias}', None, 0, None)
    return length_sec

_opener_order = {"greet": None, "antagonize": None}   # shuffled once per sequence, indexed by count

def voice_line(category, count):
    """count = how many times this gesture has now been activated (1-indexed).
    greet is a fixed sequence (hello-there -> nice-day-huh -> farewell); the
    two antagonize openers are shuffled each sequence since their order
    doesn't matter, as long as the 3rd+ activation is always the farewell.
    Returns (clip_name, length_sec)."""
    openers = GREET_OPENERS if category == "greet" else ANTAGONIZE_OPENERS
    farewell = GREET_FAREWELL if category == "greet" else ANTAGONIZE_FAREWELL
    if count == 1:
        if category == "greet":
            order = openers[:]                 # fixed order: hello-there, then nice-day-huh
        else:
            order = openers[:]
            random.shuffle(order)
        _opener_order[category] = order
    if count >= 3:
        name = farewell
    else:
        order = _opener_order[category] or openers
        name = order[(count - 1) % len(order)]
    length_sec = play_mp3(os.path.join(AUDIO_DIR, name))
    return name, length_sec

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PL = mp.tasks.vision.PoseLandmark

ANTAGONIZE_STREAK = 4        # consecutive frames required before a held pose counts (debounce)
PEACE_STREAK = 4
PUNCH_STREAK = 4
WAVE_HISTORY_SEC = 1.2       # how far back the wave detector looks
WAVE_MIN_REVERSALS = 3       # direction changes required to call it a "wave" not a raise
WAVE_MIN_AMPLITUDE = 0.35    # side-to-side travel, as a fraction of shoulder width
PUNCH_MAX_DIST = 2.2         # fist-to-palm distance allowed, as a multiple of hand size
POST_CLIP_PAUSE_SEC = 1.0    # breathing room after a clip finishes before the next
                              # trigger is allowed, on top of its real length
                              # (see voice_line) — so playback always completes
MIN_GESTURE_GAP_SEC = 5.0    # hard floor on the gap between two different gestures
                              # being recognized, even after a short clip finishes

BONES = [(PL.LEFT_SHOULDER.value, PL.RIGHT_SHOULDER.value),
         (PL.LEFT_SHOULDER.value, PL.LEFT_ELBOW.value), (PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value),
         (PL.RIGHT_SHOULDER.value, PL.RIGHT_ELBOW.value), (PL.RIGHT_ELBOW.value, PL.RIGHT_WRIST.value)]

# Standard 21-point hand topology (stable across mediapipe versions).
WRIST = 0
FINGERS = {  # name -> (mcp, pip, tip)
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
HAND_BONES = [(WRIST, mcp) for mcp, _, _ in FINGERS.values()] + \
             [(mcp, pip) for mcp, pip, _ in FINGERS.values()] + \
             [(pip, tip) for _, pip, tip in FINGERS.values()]


def xy(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y])


def is_antagonize(landmarks):
    """Bear-attack stance: both arms raised, elbows out, wrists spread wide."""
    ls, rs = xy(landmarks, PL.LEFT_SHOULDER.value), xy(landmarks, PL.RIGHT_SHOULDER.value)
    le, re = xy(landmarks, PL.LEFT_ELBOW.value), xy(landmarks, PL.RIGHT_ELBOW.value)
    lw, rw = xy(landmarks, PL.LEFT_WRIST.value), xy(landmarks, PL.RIGHT_WRIST.value)
    shoulder_width = np.linalg.norm(ls - rs) + 1e-6

    hands_up = lw[1] < ls[1] and rw[1] < rs[1]                    # smaller y = higher on screen
    elbows_up = le[1] < ls[1] + 0.05 and re[1] < rs[1] + 0.05
    spread_wide = abs(lw[0] - rw[0]) > shoulder_width * 1.3
    return hands_up and elbows_up and spread_wide


def raised_hand(landmarks):
    """Return (wrist_xy, shoulder_width) for a single raised hand, or None."""
    ls, rs = xy(landmarks, PL.LEFT_SHOULDER.value), xy(landmarks, PL.RIGHT_SHOULDER.value)
    lw, rw = xy(landmarks, PL.LEFT_WRIST.value), xy(landmarks, PL.RIGHT_WRIST.value)
    shoulder_width = np.linalg.norm(ls - rs) + 1e-6
    left_up, right_up = lw[1] < ls[1] - 0.03, rw[1] < rs[1] - 0.03
    if left_up and not right_up: return lw, shoulder_width
    if right_up and not left_up: return rw, shoulder_width
    return None


def finger_extended(landmarks, mcp, pip, tip):
    """A finger is 'up' if its tip is farther from the wrist than its PIP
    joint — a distance check, not a screen-up/down check, so it works
    regardless of how the hand is rotated."""
    w = xy(landmarks, WRIST)
    return np.linalg.norm(xy(landmarks, tip) - w) > np.linalg.norm(xy(landmarks, pip) - w) * 1.15


def is_peace_hand(landmarks):
    ext = {n: finger_extended(landmarks, mcp, pip, tip) for n, (mcp, pip, tip) in FINGERS.items()}
    return ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]


THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4

def is_thumbs_up(landmarks):
    """Four fingers curled into a fist, thumb extended and pointing up the screen."""
    fingers_folded = not any(finger_extended(landmarks, mcp, pip, tip) for mcp, pip, tip in FINGERS.values())
    thumb_extended = finger_extended(landmarks, THUMB_MCP, THUMB_IP, THUMB_TIP)
    thumb_up = xy(landmarks, THUMB_TIP)[1] < xy(landmarks, THUMB_MCP)[1] - 0.05   # smaller y = higher
    return fingers_folded and thumb_extended and thumb_up


def any_thumbs_up(hands):
    """hands: list of (real_side, landmarks). True if either hand is thumbs-up."""
    return any(is_thumbs_up(lm) for _, lm in hands)


def is_fist_hand(landmarks):
    return not any(finger_extended(landmarks, mcp, pip, tip) for mcp, pip, tip in FINGERS.values())


def is_open_palm_hand(landmarks):
    return all(finger_extended(landmarks, mcp, pip, tip) for mcp, pip, tip in FINGERS.values())


def palm_center(landmarks):
    idx = [WRIST] + [mcp for mcp, _, _ in FINGERS.values()]
    return np.mean([xy(landmarks, i) for i in idx], axis=0)


def hand_size(landmarks):
    return np.linalg.norm(xy(landmarks, FINGERS["middle"][0]) - xy(landmarks, WRIST)) + 1e-6


def both_hands_peace(hands):
    """hands: list of (real_side, landmarks). True if 2+ hands are all peace signs."""
    return len(hands) >= 2 and all(is_peace_hand(lm) for _, lm in hands)


def punch_ready(hands):
    """True if a real right-hand fist is struck close to a real left-hand open palm."""
    right = next((lm for side, lm in hands if side == "Right" and is_fist_hand(lm)), None)
    left = next((lm for side, lm in hands if side == "Left" and is_open_palm_hand(lm)), None)
    if right is None or left is None: return False
    dist = np.linalg.norm(palm_center(right) - palm_center(left))
    size = (hand_size(right) + hand_size(left)) / 2
    return dist < size * PUNCH_MAX_DIST


def is_boxer_guard(hands, pose_landmarks):
    """Both hands clenched fists, held up near the face — a boxer's guard.
    Pose and hand landmarks share the same normalized frame coordinates, so
    the nose/shoulder positions from the pose model can be compared directly
    against the hand model's fist positions."""
    if pose_landmarks is None: return False
    fists = [lm for _, lm in hands if is_fist_hand(lm)]
    if len(fists) < 2: return False

    nose = xy(pose_landmarks, PL.NOSE.value)
    ls, rs = xy(pose_landmarks, PL.LEFT_SHOULDER.value), xy(pose_landmarks, PL.RIGHT_SHOULDER.value)
    shoulder_width = np.linalg.norm(ls - rs) + 1e-6
    shoulder_y = max(ls[1], rs[1])   # smaller y = higher; this is the lower of the two shoulders

    near_face = all(np.linalg.norm(palm_center(f) - nose) < shoulder_width * 1.3 for f in fists)
    raised = all(palm_center(f)[1] < shoulder_y + 0.05 for f in fists)
    return near_face and raised


class WaveDetector:
    """Tracks a raised wrist's x position and looks for side-to-side oscillation."""
    def __init__(self):
        self.hist = deque()   # (t, x / shoulder_width)

    def update(self, hand):
        now = time.time()
        if hand is None:
            self.hist.clear()
            return False
        wrist_xy, shoulder_width = hand
        self.hist.append((now, wrist_xy[0] / shoulder_width))
        while self.hist and now - self.hist[0][0] > WAVE_HISTORY_SEC:
            self.hist.popleft()
        if len(self.hist) < 6:
            return False
        xs = [x for _, x in self.hist]
        reversals, direction = 0, 0
        for i in range(1, len(xs)):
            d = xs[i] - xs[i - 1]
            if abs(d) < 1e-3: continue
            nd = 1 if d > 0 else -1
            if direction and nd != direction: reversals += 1
            direction = nd
        amplitude = max(xs) - min(xs)
        return reversals >= WAVE_MIN_REVERSALS and amplitude >= WAVE_MIN_AMPLITUDE


def draw_skeleton(frame, landmarks, w, h):
    pts = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in {a for bone in BONES for a in bone}}
    for a, b in BONES:
        cv2.line(frame, pts[a], pts[b], (0, 255, 255), 3)
    for p in pts.values():
        cv2.circle(frame, p, 6, (0, 140, 255), -1)


def draw_hand(frame, landmarks, side, w, h):
    pts = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(21)}
    for a, b in HAND_BONES:
        cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)
    for p in pts.values():
        cv2.circle(frame, p, 3, (255, 120, 0), -1)
    cv2.putText(frame, side[0], pts[WRIST], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera found")

    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    waver = WaveDetector()
    antagonize_streak = peace_streak = punch_streak = thumbs_streak = boxer_streak = 0
    greet_count = antagonize_count = 0
    last_gesture = None
    audio_busy_until = 0.0
    t0 = time.time()

    print("Tracking arms + hands. GREET: hello wave, a two-hand peace sign, or a "
          "thumbs up. ANTAGONIZE: bear stance (arms up, elbows out, spread wide), a "
          "right fist struck into your open left palm, or a boxer's guard (both fists "
          "up near your face). r=reset counters, Esc to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.time() - t0) * 1000)
            pose_result = pose_lm.detect_for_video(mp_image, ts)
            hand_result = hand_lm.detect_for_video(mp_image, ts)

            # frame was mirrored for display, so mediapipe's Left/Right (based
            # on the mirrored image) is the opposite of the person's real hand
            hands = []
            for lm, handed in zip(hand_result.hand_landmarks, hand_result.handedness):
                reported = handed[0].category_name
                real_side = "Right" if reported == "Left" else "Left"
                hands.append((real_side, lm))
                draw_hand(frame, lm, real_side, w, h)

            bear_pose = False
            wave = False
            landmarks = None
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                draw_skeleton(frame, landmarks, w, h)
                bear_pose = is_antagonize(landmarks)
                wave = waver.update(None if bear_pose else raised_hand(landmarks))
            else:
                waver.update(None)

            punch = punch_ready(hands)
            peace = both_hands_peace(hands)
            thumbs = any_thumbs_up(hands)
            boxer = is_boxer_guard(hands, landmarks)

            antagonize_streak = antagonize_streak + 1 if (bear_pose or punch or boxer) else 0
            peace_streak = peace_streak + 1 if peace else 0
            punch_streak = punch_streak + 1 if punch else 0
            thumbs_streak = thumbs_streak + 1 if thumbs else 0
            boxer_streak = boxer_streak + 1 if boxer else 0

            gesture, reason = None, None
            if antagonize_streak >= ANTAGONIZE_STREAK:
                gesture = "antagonize"
                if punch_streak >= PUNCH_STREAK: reason = "fist bump to palm"
                elif boxer_streak >= ANTAGONIZE_STREAK: reason = "boxer guard"
                else: reason = "bear pose"
            elif wave:
                gesture, reason = "greet", "hello wave"
            elif peace_streak >= PEACE_STREAK:
                gesture, reason = "greet", "peace sign"
            elif thumbs_streak >= PEACE_STREAK:
                gesture, reason = "greet", "thumbs up"

            now = time.time()
            if gesture != last_gesture and gesture is not None and now >= audio_busy_until:
                if gesture == "greet":
                    greet_count += 1
                    clip, length_sec = voice_line("greet", greet_count)
                else:
                    antagonize_count += 1
                    clip, length_sec = voice_line("antagonize", antagonize_count)
                audio_busy_until = now + max(length_sec + POST_CLIP_PAUSE_SEC, MIN_GESTURE_GAP_SEC)
                print(f"EMOTION: {gesture.upper()} ({reason}) -> {clip} ({length_sec:.1f}s)")
            last_gesture = gesture

            color = {"greet": (0, 255, 0), "antagonize": (0, 0, 255)}.get(gesture, (200, 200, 200))
            cv2.putText(frame, gesture or "...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"greet x{greet_count}  antagonize x{antagonize_count}  (r=reset)",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Gestures", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('r'):
                greet_count = antagonize_count = 0
                last_gesture = None   # so the very next gesture re-triggers a fresh opener line
                audio_busy_until = 0.0
                print("-- counters reset, sequence will repeat from the opener lines --")
    finally:
        pose_lm.close()
        hand_lm.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
