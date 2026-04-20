"""
Hand Gesture Recognition  |  Task 4 - Prodigy ML Internship
-------------------------------------------------------------
Counts extended fingers (0-5) in real-time using MediaPipe
HandLandmarker with robust geometric rules.

Key fixes over v1:
  • Thumb: compare TIP vs MCP (landmark 2) on the x-axis — much more
    robust than TIP vs IP which fails on tilted hands.
  • Handedness: after cv2.flip the labels are swapped — "Right" in the
    result actually means the user's Left hand. We invert accordingly.
  • Timestamps: use real wall-clock milliseconds instead of a fixed +33ms
    drift so VIDEO running mode stays perfectly in sync.
  • Temporal smoothing: a small deque of recent counts is used to prevent
    flickering between values.
  • UI overhaul: cleaner layout, per-finger pip icons, confidence bar.

Install : pip install mediapipe opencv-python numpy
Run     : python Model.py          (press Q to quit)
"""

import os, time, urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── Model download ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker"
              "/hand_landmarker/float16/latest/hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ── Landmark indices ────────────────────────────────────────────────────────────
#   MediaPipe 21-point hand schema
#   Thumb  : CMC=1  MCP=2  IP=3   TIP=4
#   Index  : MCP=5  PIP=6  DIP=7  TIP=8
#   Middle : MCP=9  PIP=10 DIP=11 TIP=12
#   Ring   : MCP=13 PIP=14 DIP=15 TIP=16
#   Pinky  : MCP=17 PIP=18 DIP=19 TIP=20

FINGER_TIPS  = [8, 12, 16, 20]   # index → pinky TIP indices
FINGER_PIPS  = [6, 10, 14, 18]   # corresponding PIP (2nd knuckle) indices

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

# BGR colors per count
COUNT_COLORS = {
    0: (90,  90,  90),
    1: (60, 220,  60),
    2: (0,  170, 255),
    3: (30, 140, 255),
    4: (200,  60, 200),
    5: (40,  80, 255),
}

# ── Finger counting (corrected) ─────────────────────────────────────────────────
def count_fingers(lm, is_user_right: bool) -> tuple[int, list[bool]]:
    """
    Returns (total_count, [thumb_up, idx_up, mid_up, ring_up, pinky_up]).

    lm            -- list of NormalizedLandmark (21 pts, 0-1 coords).
    is_user_right -- True if this is the user's RIGHT hand (after accounting
                     for the mirror flip already applied to the frame).

    Thumb rule  : TIP (4) must be clearly to the LEFT of MCP (2) for a right
                  hand (further from palm centre) — and vice-versa for left.
                  Using MCP instead of IP gives a much wider, more stable gap.

    Finger rule : TIP.y < PIP.y  (y increases downward, so tip above pip = up).
                  We add a small hysteresis gap (2 % of image height) to avoid
                  flickering at borderline positions.
    """
    up = []
    HYSTERESIS = 0.02   # normalised units (~2 % of frame height)

    # ── Thumb ──
    tip_x = lm[4].x
    mcp_x = lm[2].x   # MCP = knuckle 2 (base of thumb)
    if is_user_right:
        # Right hand (mirrored frame): tip should be LEFT of MCP when extended
        up.append(tip_x < mcp_x - HYSTERESIS)
    else:
        # Left hand (mirrored frame): tip should be RIGHT of MCP
        up.append(tip_x > mcp_x + HYSTERESIS)

    # ── Four fingers ──
    for tip_i, pip_i in zip(FINGER_TIPS, FINGER_PIPS):
        up.append(lm[tip_i].y < lm[pip_i].y - HYSTERESIS)

    return sum(up), up


# ── Skeleton connections ────────────────────────────────────────────────────────
CONNS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm arch
]

# ── Drawing ─────────────────────────────────────────────────────────────────────
def draw_hand(frame, lm, count, up_flags, is_right_label: str):
    h, w = frame.shape[:2]
    pts  = [(int(l.x * w), int(l.y * h)) for l in lm]

    # — skeleton —
    for a, b in CONNS:
        cv2.line(frame, pts[a], pts[b], (30, 190, 30), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        is_tip = i in (4, 8, 12, 16, 20)
        color  = (0, 255, 100) if is_tip else (220, 220, 220)
        radius = 7 if is_tip else 4
        cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)


def draw_ui(frame, hands_data):
    """
    hands_data : list of (count, up_flags, hand_label_str)
    Renders the top header bar and bottom finger-status bar.
    """
    h, w = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────────────────────────────
    HEADER_H = 80
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_H), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    if not hands_data:
        cv2.putText(frame, "Show your hand to the camera",
                    (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2, cv2.LINE_AA)
        return

    # If two hands, show combined total on right
    if len(hands_data) > 1:
        total = sum(d[0] for d in hands_data)
        cv2.putText(frame, f"Total: {total}",
                    (w - 180, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 60), 2, cv2.LINE_AA)

    # Primary (first) hand in header
    count, up_flags, hand_label = hands_data[0]
    color = COUNT_COLORS[count]
    label = f"{LABELS[count]}  ({count})"
    cv2.putText(frame, label, (14, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.55, color, 3, cv2.LINE_AA)
    cv2.putText(frame, hand_label + " Hand",
                (14, HEADER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (160, 160, 160), 1, cv2.LINE_AA)

    # ── Bottom finger-status bar ──────────────────────────────────────────────
    BAR_H  = 54
    bar_y0 = h - BAR_H
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y0), (w, h), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Use the first hand's up_flags for individual display
    col_w  = w // 5
    for i, (name, status) in enumerate(zip(FINGER_NAMES, up_flags)):
        cx = i * col_w + col_w // 2
        fc = (0, 220, 80) if status else (70, 70, 70)

        # Circle pip indicator
        pip_y  = bar_y0 + 14
        cv2.circle(frame, (cx, pip_y), 9, fc, -1, cv2.LINE_AA)
        if status:
            cv2.circle(frame, (cx, pip_y), 9, (255, 255, 255), 1, cv2.LINE_AA)

        # Label
        text_x = cx - (len(name) * 7) // 2
        cv2.putText(frame, name, (text_x, bar_y0 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, fc, 1, cv2.LINE_AA)


# ── Temporal smoother ───────────────────────────────────────────────────────────
class CountSmoother:
    """
    Majority-vote over the last N frames to suppress single-frame glitches.
    """
    def __init__(self, window: int = 6):
        self.buf = deque(maxlen=window)

    def update(self, count: int) -> int:
        self.buf.append(count)
        return max(set(self.buf), key=self.buf.count)   # mode


# ── Main webcam loop ────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible."); return

    # Try to boost camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    opts = HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.5,
    )

    smoothers: dict[int, CountSmoother] = {}   # one smoother per hand slot
    start_ms = int(time.time() * 1000)
    print("Webcam active — show your hand and extend fingers. Press Q to quit.")

    with HandLandmarker.create_from_options(opts) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed, retrying..."); continue

            # Mirror so user's right hand appears on the right side of screen
            frame = cv2.flip(frame, 1)

            # Real wall-clock timestamp (milliseconds) for VIDEO mode
            ts_ms = int(time.time() * 1000) - start_ms

            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result  = detector.detect_for_video(mp_img, ts_ms)

            hands_data = []

            if result.hand_landmarks:
                for slot, (lm, handedness) in enumerate(zip(result.hand_landmarks,
                                                             result.handedness)):
                    # After flip, MediaPipe's "Right" label = user's LEFT hand.
                    mp_label     = handedness[0].display_name  # "Right" or "Left"
                    is_user_right = (mp_label == "Left")        # inverted post-flip
                    hand_str      = "Right" if is_user_right else "Left"

                    raw_count, up_flags = count_fingers(lm, is_user_right)

                    # Smooth count
                    if slot not in smoothers:
                        smoothers[slot] = CountSmoother(window=6)
                    smooth_count = smoothers[slot].update(raw_count)

                    draw_hand(frame, lm, smooth_count, up_flags, hand_str)
                    hands_data.append((smooth_count, up_flags, hand_str))
            else:
                smoothers.clear()   # reset smoothers when no hand visible

            draw_ui(frame, hands_data)

            # FPS overlay (top-right corner)
            cv2.putText(frame, f"ts:{ts_ms//1000}s",
                        (frame.shape[1] - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

            cv2.imshow("Finger Counter  |  Task 4  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
