import cv2
import torch
import numpy as np
import mediapipe as mp
import sys, os
from collections import deque

sys.path.append(os.path.abspath("."))

from src.models.elbow_lstm import ElbowLSTM
from src.features.elbow_features import build_features

SEQ_LEN = 128

# -----------------------------
# RESIZE
# -----------------------------
def resize_sequence(features, target_len=128):
    if len(features) == 0:
        return np.zeros((target_len, 8), dtype=np.float32)

    if len(features) < target_len:
        pad = np.repeat(features[-1:], target_len - len(features), axis=0)
        return np.vstack((features, pad)).astype(np.float32)

    idx = np.linspace(0, len(features)-1, target_len).astype(int)
    return features[idx].astype(np.float32)


# -----------------------------
# 🔥 BALANCED CLASSIFIER
# -----------------------------
def classify_rep(pred, angles, wrist_path):
    angles = np.array(angles)

    range_motion = np.max(angles) - np.min(angles)
    peak = np.max(angles)
    valley = np.min(angles)

    velocity = np.diff(angles)
    smoothness = np.std(velocity)

    wrist_path = np.array(wrist_path)
    wrist_movement = np.linalg.norm(wrist_path[1:] - wrist_path[:-1], axis=1).mean()

    # -----------------------------
    # 🔥 HARD FAIL CONDITIONS (KEY FIX)
    # -----------------------------
    penalty = 0

    if range_motion < 25:     # half rep
        penalty += 0.25

    if peak < 125:           # not extending
        penalty += 0.20

    if smoothness > 20:      # jerky
        penalty += 0.15

    if wrist_movement > 0.04:  # unstable arm
        penalty += 0.15

    # -----------------------------
    # 🔥 NORMAL SCORE
    # -----------------------------
    range_score = min(1, range_motion / 70)
    smooth_score = max(0, 1 - smoothness / 25)

    base_score = (
        0.6 * pred +
        0.2 * range_score +
        0.2 * smooth_score
    )

    # -----------------------------
    # 🔥 FINAL SCORE (IMPORTANT)
    # -----------------------------
    final_score = base_score - penalty

    return final_score, range_motion, smoothness, peak, wrist_movement


# -----------------------------
# MODEL
# -----------------------------
model = ElbowLSTM(input_size=8)
model.load_state_dict(
    torch.load("models/elbow_lstm.pth", map_location="cpu", weights_only=True)
)
model.eval()


# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# -----------------------------
# 🔥 STABLE SMOOTHING
# -----------------------------
ema_angle = None
alpha = 0.25   # slightly higher → less lag

angle_buffer = deque(maxlen=5)

prev_angle = None
state = "Stable"

# hysteresis thresholds
UP_THRESH = 0.8
DOWN_THRESH = -0.8
STABLE_THRESH = 0.3

# rep tracking
rep_active = False
rep_angles = []
rep_rows = []
wrist_path = []
rep_count = 0

# output
final_label = "Waiting..."
final_color = (255, 255, 255)
feedback = ""

print("Press Q to exit")


def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(cosine))


def extract_row(landmarks):
    row = {}
    for i, lm in enumerate(landmarks):
        row[f"landmark_{i}_x"] = lm.x
        row[f"landmark_{i}_y"] = lm.y
        row[f"landmark_{i}_z"] = lm.z
    return row


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        shoulder = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
        elbow = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
        wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z]

        raw_angle = get_angle(shoulder, elbow, wrist)

        # 🔥 EMA smoothing
        if ema_angle is None:
            ema_angle = raw_angle
        else:
            ema_angle = alpha * raw_angle + (1 - alpha) * ema_angle

        angle_buffer.append(ema_angle)
        angle = np.mean(angle_buffer)

        # -----------------------------
        # 🔥 STABLE STATE MACHINE
        # -----------------------------
        if prev_angle is None:
            delta = 0
        else:
            delta = angle - prev_angle

        prev_angle = angle

        # hysteresis logic
        if delta > UP_THRESH:
            state = "Extending"
        elif delta < DOWN_THRESH:
            state = "Flexing"
        elif abs(delta) < STABLE_THRESH:
            state = "Stable"

        # -----------------------------
        # 🔥 REP DETECTION (RELAXED)
        # -----------------------------
        if state == "Flexing" and not rep_active:
            rep_active = True
            rep_angles = []
            rep_rows = []
            wrist_path = []

        if rep_active:
            rep_angles.append(angle)
            rep_rows.append(extract_row(landmarks))
            wrist_path.append(wrist)

        if state == "Extending" and rep_active and len(rep_angles) > 20:

            if np.max(rep_angles) - np.min(rep_angles) > 20:   # relaxed

                rep_active = False
                rep_count += 1

                features, _ = build_features(rep_rows, fps=30, angle_series=rep_angles)
                features = resize_sequence(features)

                x = torch.tensor(features).unsqueeze(0).float()

                with torch.no_grad():
                    pred = torch.sigmoid(model(x)).item()

                score, r, s, p, w = classify_rep(pred, rep_angles, wrist_path)

                # 🔥 BALANCED DECISION
                if score > 0.58:
                    final_label = f"CORRECT ({score:.2f})"
                    final_color = (0, 255, 0)
                else:
                    final_label = f"INCORRECT ({score:.2f})"
                    final_color = (0, 0, 255)

                # 🔥 FEEDBACK
                if r < 25:
                    feedback = "Increase range"
                elif p < 130:
                    feedback = "Extend fully"
                elif s > 18:
                    feedback = "Move smoother"
                elif w > 0.03:
                    feedback = "Keep arm stable"
                else:
                    feedback = "Good form"

        # -----------------------------
        # UI
        # -----------------------------
        cv2.putText(frame, f"Angle: {int(angle)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, final_label, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, final_color, 3)

        cv2.putText(frame, f"State: {state}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

        cv2.putText(frame, f"Reps: {rep_count}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, feedback, (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Physio AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()