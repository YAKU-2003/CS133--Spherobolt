import time, logging, functools, threading, wave, math, struct, winsound, os
import cv2, numpy as np

_orig = functools.partialmethod._make_unbound_method
def _compat(self):
    m = _orig(self); m._partialmethod = self; return m
functools.partialmethod._make_unbound_method = _compat

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

logging.getLogger().setLevel(logging.CRITICAL)
ORANGE=Color(255,40,0); BLACK=Color(0,0,0)
# Magenta: what the BOLT lights up while it needs to be visually tracked.
# Orange (hue~5) sat too close to the red/skin-tone zone and to washed-out
# highlights — bright warm LEDs commonly clip toward white under a webcam's
# auto-exposure, losing saturation. Magenta is rare indoors, holds its
# saturation better when overexposed, and is far from both red and MARK's
# yellow hue range, so it's much less likely to be lost or false-matched.
TRACK=Color(255,0,255)

CAM_INDEX = None
GRID = 6

# Ball = the BOLT itself, tracked by the magenta color it displays (hue ~150).
BALL_LO=np.array([135,110,110]); BALL_HI=np.array([165,255,255])
# Mark = the target to chase: a yellow smiley stress ball (hue ~30). Wider
# hue window and lower S/V floor than the ball's — it's plain plastic under
# room light (not a bright LED), so it's dimmer/less saturated and its hue
# shifts more with lighting color temperature.
MARK_LO=np.array([15,60,80]);    MARK_HI=np.array([40,255,255])

SMILE=["........",".XX..XX.",".XX..XX.","........","X......X","X......X",".XXXXXX.","..XXXX.."]
RESTING=["..XXXX..",".X....X.","X..XX..X","X.X..X.X","X..XX..X",".X.XX.X.","...XX...","..XXXX.."]

def make_jingle(path="victory.wav"):
    sr=44100; notes=[523,659,784,1047]; dur=0.16; fr=bytearray()
    for f in notes:
        for s in range(int(sr*dur)):
            t=s/sr; env=min(1,8*t)*max(0,1-t/dur)
            fr+=struct.pack('<h',int(env*math.sin(2*math.pi*f*t)*22000))
    for s in range(int(sr*0.4)):
        t=s/sr; env=max(0,1-t/0.4)
        v=env*(0.5*math.sin(2*math.pi*1047*t)+0.3*math.sin(2*math.pi*1319*t))
        fr+=struct.pack('<h',int(v*20000))
    with wave.open(path,'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(bytes(fr))
    return os.path.abspath(path)
JINGLE=make_jingle()
def play_jingle(): winsound.PlaySound(JINGLE, winsound.SND_FILENAME|winsound.SND_ASYNC)

PHI0=0.0; CHIRALITY=1

# ---- threaded frame reader: main loop always gets the newest frame ---------
class FrameGrabber:
    """Reads the stream on a background thread and keeps only the newest frame,
    so the main loop never processes OpenCV's queued backlog. (from the LEGO rig)"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source!r}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._lock = threading.Lock()
        self._frame = None; self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05); continue
            with self._lock:
                self._frame = frame
    def read(self):                       # drop-in for cv2.VideoCapture.read()
        with self._lock:
            return (self._frame is not None), self._frame
    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()

def select_camera_index():
    """Return the camera index to use (picks the DroidCam feed if multiple)."""
    if CAM_INDEX is not None: return CAM_INDEX
    working=[]
    for i in range(5):
        c=cv2.VideoCapture(i); ok,_=c.read()
        if ok: working.append(i)
        c.release()
    if not working: raise RuntimeError("No camera — is DroidCam running?")
    if len(working)==1: print("cam",working[0]); return working[0]
    for i in working:
        c=cv2.VideoCapture(i); t=time.time()
        while time.time()-t<3:
            ok,f=c.read()
            if ok:
                cv2.putText(f,f"idx {i}: SPACE=use, other=next",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                cv2.imshow("pick camera",f)
            k=cv2.waitKey(1)&0xFF
            if k==32: c.release(); cv2.destroyWindow("pick camera"); return i
            elif k!=255: break
        c.release()
    cv2.destroyWindow("pick camera"); return working[-1]

def draw(b,g,c=ORANGE):
    b.set_matrix_fill(0,0,7,7,BLACK)
    for r,l in enumerate(g):
        for col,ch in enumerate(l):
            if ch=="X": b.set_matrix_pixel(col,r,c)

BALL_MIN_AREA = 10   # px, after dilation — the LED matrix is small and sparse
MARK_MIN_AREA = 40   # px, after dilation — a stress ball held at arm's length
                      # can still be much smaller than one 6x6 grid cell (~8500px)

def find_blob(mask, ch, cw, min_area, dilate_iters=2):
    """Contour-based centroid of the largest blob in `mask`. A grid-fill-fraction
    test (like the old `best()`) dilutes a small object across a whole grid
    cell and never crosses threshold — contours find it regardless of how
    small the actual lit/colored area is."""
    grown = cv2.dilate(mask, None, iterations=dilate_iters)
    cnts,_ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area: return None
    (cx,cy),_ = cv2.minEnclosingCircle(c)
    cx,cy = int(cx),int(cy)
    h,w = mask.shape[:2]
    r,col = min(cy//ch, h//ch-1), min(cx//cw, w//cw-1)
    return (r,col,cx,cy)

def cells(frame):
    """Return (ball_cell, mark_cell), each (row,col,cx,cy) or None. One HSV pass."""
    h,w=frame.shape[:2]; ch,cw=h//GRID,w//GRID
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    bmask=cv2.inRange(hsv,BALL_LO,BALL_HI)
    mmask=cv2.inRange(hsv,MARK_LO,MARK_HI)
    ball=find_blob(bmask,ch,cw,BALL_MIN_AREA,dilate_iters=3)   # matrix pattern has gaps
    mark=find_blob(mmask,ch,cw,MARK_MIN_AREA,dilate_iters=1)   # solid ball, just clean noise
    return ball,mark

def draw_grid(frame, ball, mark):
    h,w=frame.shape[:2]; ch,cw=h//GRID,w//GRID
    for i in range(1,GRID):
        cv2.line(frame,(i*cw,0),(i*cw,h),(60,60,60),1)
        cv2.line(frame,(0,i*ch),(w,i*ch),(60,60,60),1)
    for cell,color in ((ball,(255,150,0)),(mark,(0,255,0))):
        if cell:
            r,c,_,_=cell
            cv2.rectangle(frame,(c*cw,r*ch),((c+1)*cw,(r+1)*ch),color,3)

def calibrate(b,cam):
    global PHI0,CHIRALITY
    b.set_main_led(TRACK); time.sleep(0.5)
    def ballpt():
        best_area=0.0
        for _ in range(30):
            ok,f=cam.read()
            if ok:
                bc,_=cells(cv2.flip(f,1))
                if bc: return (bc[2],bc[3])
                # track the largest orange blob seen, for diagnostics on failure
                hsv=cv2.cvtColor(cv2.flip(f,1),cv2.COLOR_BGR2HSV)
                mask=cv2.dilate(cv2.inRange(hsv,BALL_LO,BALL_HI),None,iterations=3)
                cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if cnts: best_area=max(best_area,cv2.contourArea(max(cnts,key=cv2.contourArea)))
            time.sleep(0.05)
        print(f"  no magenta blob found (largest was {best_area:.0f}px, "
              f"need >{BALL_MIN_AREA}px) — press 'm' to check the mask")
        return None
    p0=ballpt()
    if not p0: return False
    b.roll(0,70,0.7); b.set_speed(0); time.sleep(0.4); p1=ballpt()
    if not p1 or abs(p1[0]-p0[0])+abs(p1[1]-p0[1])<10: return False
    PHI0=np.degrees(np.arctan2(p1[1]-p0[1],p1[0]-p0[0]))
    b.roll(90,70,0.7); b.set_speed(0); time.sleep(0.4); p2=ballpt()
    if not p2: return False
    phi90=np.degrees(np.arctan2(p2[1]-p1[1],p2[0]-p1[0]))
    d=((phi90-PHI0+180)%360)-180; CHIRALITY=1 if d>0 else -1
    print(f"Calibrated PHI0={PHI0:.0f} chir={CHIRALITY}"); return True

def heading_to(ball,tgt):
    des=np.degrees(np.arctan2(tgt[1]-ball[1],tgt[0]-ball[0]))
    return int((CHIRALITY*(des-PHI0))%360)

def spot_reaction(b):
    """Personality beat: BOLT 'notices' the yellow stress ball the moment it
    comes into view — a quick excited wiggle + the victory jingle, before it
    settles in to actually chase it."""
    draw(b,SMILE,TRACK); play_jingle()   # stay magenta — the camera is mid-track, don't blind it
    b.spin(200,0.3); time.sleep(0.05); b.spin(-200,0.3)
    b.set_speed(0); draw(b,RESTING,TRACK)

def follow(b,cam,timeout=20):
    b.set_main_led(TRACK); draw(b,RESTING,TRACK); end=time.time()+timeout
    spotted=False; last_log=0; drive_calls=0
    while time.time()<end:
        ok,f=cam.read()
        if not ok:
            time.sleep(0.02); continue
        f=cv2.flip(f,1)
        ball,mark=cells(f)
        draw_grid(f,ball,mark); cv2.imshow("Pluto",f); cv2.waitKey(1)
        if mark and not spotted:
            spotted=True
            spot_reaction(b)
            continue
        if ball and mark:
            if abs(ball[0]-mark[0])<=1 and abs(ball[1]-mark[1])<=1:
                b.set_speed(0); draw(b,SMILE,TRACK); play_jingle()   # still magenta through the spin
                time.sleep(0.3); b.spin(360,0.6); time.sleep(1.0)
                b.set_main_led(ORANGE); draw(b,RESTING); return True   # done chasing — back to idle orange
            h=heading_to((ball[2],ball[3]),(mark[2],mark[3]))
            drive_calls+=1
            if time.time()-last_log>0.5:
                print(f"  ball@{ball[:2]} mark@{mark[:2]} -> heading {h} (PHI0={PHI0:.0f} chir={CHIRALITY})")
                last_log=time.time()
            b.roll(h,55,0.2); b.set_speed(0); time.sleep(0.15)   # gentler + settle for lag
        elif spotted and time.time()-last_log>0.5:
            print(f"  lost track — ball {'seen' if ball else 'LOST'}, mark {'seen' if mark else 'LOST'}")
            last_log=time.time()
        if (cv2.waitKey(1)&0xFF)==27: break
    if spotted and drive_calls==0:
        print("  never had both ball and marker at once — check the mask ('m') for both colors")
    b.set_speed(0); b.set_main_led(ORANGE); draw(b,RESTING); return False

def introduce(b):
    try: b.reset_aim()
    except Exception: pass
    draw(b,RESTING); b.set_main_led(ORANGE); b.set_heading(0); time.sleep(0.3)
    b.roll(0,110,1.0); b.set_speed(0); time.sleep(0.3)
    b.set_heading(45); time.sleep(0.5); b.set_heading(315); time.sleep(0.5)
    b.set_heading(0); b.set_speed(0); draw(b,RESTING)

def main():
    # NOTE: scanner.find_BOLT() MUST run before opening the camera. Opening the
    # camera first makes OpenCV's Media Foundation/DirectShow backend init COM
    # as STA on this thread; bleak's WinRT scanner then fails with "Thread is
    # configured for Windows GUI but callbacks are not working" (needs MTA).
    # Do not reorder these.
    print("Connecting — shake BOLT awake...")
    toy=None
    for attempt in range(1,5):
        try:
            toy=scanner.find_BOLT(timeout=7.0); break
        except scanner.ToyNotFoundError:
            print(f"  not found yet (attempt {attempt}/4) — "
                  "make sure it's awake (shake it), Bluetooth is on, and it's not connected in the Sphero app")
    if toy is None:
        raise scanner.ToyNotFoundError(
            "Could not find the BOLT after 4 scans. Check it's charged/awake, "
            "Bluetooth is on, and no other app (e.g. Sphero Edu) is already connected to it.")
    cam=FrameGrabber(select_camera_index())
    for _ in range(60):                     # wait for the first frame
        if cam.read()[0]: break
        time.sleep(0.05)
    with SpheroEduAPI(toy) as b:
        # SpheroEduAPI.__enter__ turns collision detection ON by default. Real
        # collisions then fire notify packets that crash a background thread
        # in this spherov2 version (struct.error: unpack requires 18 bytes).
        # Send an actual disable command to the robot instead of a no-op patch.
        try:
            from spherov2.commands.sensor import CollisionDetectionMethods
            toy.configure_collision_detection(CollisionDetectionMethods.NO_COLLISION_DETECTION, 0,0,0,0,0)
        except Exception: pass
        try: b.reset_aim()
        except Exception: pass
        print("Say hi, Pluto!"); introduce(b)   # personality beat: show orange face + a little intro roll on boot
        print("Calibrating..."); print("OK" if calibrate(b,cam) else "CALIB FAILED")
        print("Keys: f=follow  i=introduce  c=recalib  m=cycle mask view  q=quit")
        mask_view=0   # 0=normal  1=ball(magenta) mask  2=mark(yellow) mask
        while True:
            ok,f=cam.read()
            if not ok:
                if cv2.waitKey(5)&0xFF==ord('q'): break
                continue
            f=cv2.flip(f,1)
            if mask_view:
                hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
                lo,hi,label=(BALL_LO,BALL_HI,"BALL (magenta)") if mask_view==1 else (MARK_LO,MARK_HI,"MARK (yellow)")
                m=cv2.inRange(hsv,lo,hi)
                m=cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)
                cv2.putText(m,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                cv2.imshow("Pluto",m)
            else:
                ball,mark=cells(f)
                draw_grid(f,ball,mark)
                cv2.putText(f,"f=follow i=intro c=recalib m=mask q=quit",(10,25),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
                cv2.imshow("Pluto",f)
            k=cv2.waitKey(1)&0xFF
            if   k==ord('f'): follow(b,cam)
            elif k==ord('i'): introduce(b)
            elif k==ord('c'): calibrate(b,cam)
            elif k==ord('m'): mask_view=(mask_view+1)%3
            elif k==ord('q') or k==27: break
        cam.close(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()