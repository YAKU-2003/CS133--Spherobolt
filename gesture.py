import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# fingertip landmark ids and the joint below each (PIP), for open/closed test
TIPS = [8, 12, 16, 20]       # index, middle, ring, pinky tips
PIPS = [6, 10, 14, 18]       # the joint below each tip

def fingers_folded(lm):
    """Count how many of the 4 fingers are curled (tip below its PIP joint)."""
    folded = 0
    for tip, pip in zip(TIPS, PIPS):
        # in image coords, larger y = lower on screen; folded finger's tip sits below its joint
        if lm.landmark[tip].y > lm.landmark[pip].y:
            folded += 1
    return folded

def detect_gesture(lm):
    if fingers_folded(lm) == 4:      # all four fingers curled = fist
        return "fistbump"
    return None

def main():
    cap = cv2.VideoCapture(0)        # 0 = default webcam
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)   # mirror, feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = None
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(lm)

        if gesture and gesture != last:
            print("GESTURE:", gesture)      # later: send to Pi
        last = gesture

        label = gesture if gesture else "..."
        cv2.putText(frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        cv2.imshow("Pluto Vision", frame)
        if cv2.waitKey(1) & 0xFF == 27:      # Esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()