import time, threading, logging, functools
import cv2, numpy as np
hc
# --- Python 3.14 compat shim (your fix) ---
_orig = functools.partialmethod._make_unbound_method
def _compat(self):
    m = _orig(self); m._partialmethod = self; return m
functools.partialmethod._make_unbound_method = _compat

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

logging.getLogger().setLevel(logging.CRITICAL)
ORANGE = Color(255,40,0); BLUE = Color(0,90,255); RED = Color(255,0,0); BLACK = Color(0,0,0)

# HSV range for the BLUE tracking glow (tune if needed)
LOWER = np.array([95, 80, 80]); UPPER = np.array([130,255,255])

RESTING = ["..XXXX..", ".X....X.", "X..XX..X", "X.X..X.X",
           "X..XX..X", ".X.XX.X.", "...XX...", "..XXXX.."]
SMILE   = ["........", ".XX..XX.", ".XX..XX.", "........",
           "X......X", "X......X", ".XXXXXX.", "..XXXX.."]

# calibration results (filled at startup)
PHI0 = 0.0       # image angle (deg) produced by robot heading 0
CHIRALITY = 1    # +1 or -1

def draw(b,g,c=ORANGE):
    b.set_matrix_fill(0,0,7,7,BLACK)
    for r,l in enumerate(g):
        for col,ch in enumerate(l):
            if ch=="X": b.set_matrix_pixel(col,r,c)

def find_ball(frame):
    """Return (x,y,radius) of the blue glow, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    mask = cv2.erode(mask, None, 2); mask = cv2.dilate(mask, None, 2)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 80: return None
    (x,y),r = cv2.minEnclosingCircle(c)
    return (int(x),int(y),int(r))

def grab_ball(cap, tries=8):
    for _ in range(tries):
        ok,f = cap.read()
        if not ok: continue
        f = cv2.flip(f,1); p = find_ball(f)
        if p: return p[:2]
        time.sleep(0.03)
    return None

def calibrate(b, cap):
    global PHI0, CHIRALITY
    b.set_main_led(BLUE); time.sleep(0.5)
    p0 = grab_ball(cap)
    if not p0: return False
    b.roll(0,70,0.6); b.set_speed(0); time.sleep(0.4)
    p1 = grab_ball(cap)
    if not p1: return False
    v0 = (p1[0]-p0[0], p1[1]-p0[1])
    if abs(v0[0])+abs(v0[1]) < 8: return False
    PHI0 = np.degrees(np.arctan2(v0[1], v0[0]))
    # second probe for chirality
    b.roll(90,70,0.6); b.set_speed(0); time.sleep(0.4)
    p2 = grab_ball(cap)
    if not p2: return False
    v1 = (p2[0]-p1[0], p2[1]-p1[1])
    phi90 = np.degrees(np.arctan2(v1[1], v1[0]))
    d = ((phi90 - PHI0 + 180) % 360) - 180
    CHIRALITY = 1 if d > 0 else -1
    print(f"Calibrated: PHI0={PHI0:.0f}deg chirality={CHIRALITY}")
    return True

def heading_to(ball, target):
    desired = np.degrees(np.arctan2(target[1]-ball[1], target[0]-ball[0]))
    return int((CHIRALITY*(desired - PHI0)) % 360)

def home_to(b, cap, target, arrive=45, timeout=10):
    b.set_main_led(BLUE)
    end = time.time()+timeout
    while time.time() < end:
        ok,f = cap.read()
        if not ok: break
        f = cv2.flip(f,1); ball = find_ball(f)
        cv2.circle(f, target, arrive, (0,255,0), 2)
        if ball:
            cv2.circle(f, ball[:2], ball[2], (255,150,0), 2)
            err = np.hypot(target[0]-ball[0], target[1]-ball[1])
            if err < arrive:
                b.set_speed(0); draw(b,SMILE)
                cv2.imshow("Pluto Servo", f); cv2.waitKey(1)
                time.sleep(0.4); b.spin(360,0.6)      # arrived: happy
                time.sleep(0.8); b.set_main_led(ORANGE); draw(b,RESTING)
                return True
            h = heading_to(ball[:2], target)
            spd = int(np.clip(err*0.35, 40, 110))
            b.roll(h, spd, 0.25); b.set_speed(0)
        cv2.imshow("Pluto Servo", f)
        if (cv2.waitKey(1)&0xFF)==27: break
    b.set_speed(0); b.set_main_led(ORANGE); draw(b,RESTING)
    return False

def main():
    print("Connecting — shake BOLT awake...")
    b_toy = scanner.find_BOLT()
    with SpheroEduAPI(b_toy) as b:
        try:
            from spherov2.commands.sensor import Sensor
            Sensor.configure_collision_detection = lambda *a,**k: None
        except Exception: pass
        try: b.reset_aim()
        except Exception: pass
        b.set_main_led(BLUE); draw(b,RESTING)

        cap = cv2.VideoCapture(0)
        print("Calibrating — keep the ball in view...")
        ok = calibrate(b, cap)
        print("Calibration OK" if ok else "Calibration FAILED — check camera sees the blue ball / lighting")

        print("Keys: h=home to center  c=recalibrate  q=quit")
        while True:
            ret,f = cap.read()
            if not ret: break
            f = cv2.flip(f,1); h,w = f.shape[:2]
            target = (w//2, h//2)
            ball = find_ball(f)
            cv2.circle(f, target, 45, (0,255,0), 2)
            cv2.putText(f,"press h to home",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            if ball: cv2.circle(f, ball[:2], ball[2], (255,150,0), 2)
            cv2.imshow("Pluto Servo", f)
            k = cv2.waitKey(1)&0xFF
            if   k==ord('h'): home_to(b, cap, target)
            elif k==ord('c'): calibrate(b, cap)
            elif k==ord('q') or k==27: break
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()