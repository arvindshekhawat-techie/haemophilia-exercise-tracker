"""Real-time elbow inference — FINAL (STABLE + CORRECT + BALANCED SCORE)"""

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.features.elbow_features import build_features
from src.inference.inference import ElbowInference
from src.pose.pose_extraction import extract_frame_landmarks
from src.training.dataset import SEQUENCE_LENGTH


# ================== TUNING ==================
ANGLE_EMA_ALPHA = 0.25
ANGLE_DEADZONE = 1.0

CALIBRATION_FRAMES = 40

MIN_REP_FRAMES = 10
REP_COOLDOWN = 6

DIRECTION_SMOOTH = 4

# EARLY STATE
UP_TRIGGER_OFFSET = 20
DOWN_TRIGGER_OFFSET = 12

# FORM (RELATIVE)
GOOD_RATIO = 0.55
BAD_RATIO = 0.35
# ===========================================


class LiveRepTracker:
    def __init__(self):
        self.angle_series = deque(maxlen=SEQUENCE_LENGTH)
        self.rows = deque(maxlen=SEQUENCE_LENGTH)
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)

        self.ema_angle = None
        self.last_angle = None
        self.prev_angle = None

        self.direction_buffer = deque(maxlen=DIRECTION_SMOOTH)
        self.direction = None

        self.state = "DOWN"

        # Calibration
        self.calibration = []
        self.calibrated = False
        self.min_angle = None
        self.max_angle = None

        self.up_trigger = None
        self.down_trigger = None

        # Rep tracking
        self.rep_count = 0
        self.rep_frames = 0
        self.cooldown = 0

        self.rep_angles = []
        self.rep_velocities = []

        # Model (optional)
        self.last_prob = 0.5

        # Output
        self.form = "WAITING"
        self.last_rep_score = None
        self.last_rep_status = "WAITING"

    # ================= ANGLE =================
    def smooth_angle(self, shoulder, elbow, wrist):
        a, b, c = np.array(shoulder), np.array(elbow), np.array(wrist)

        ba = a - b
        bc = c - b

        angle = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
                    -1, 1
                )
            )
        )

        angle = float(np.clip(angle, 40, 170))

        if self.ema_angle is None:
            self.ema_angle = angle
        else:
            self.ema_angle = ANGLE_EMA_ALPHA * angle + (1 - ANGLE_EMA_ALPHA) * self.ema_angle

        if self.last_angle is not None and abs(self.ema_angle - self.last_angle) < ANGLE_DEADZONE:
            return self.last_angle

        self.last_angle = self.ema_angle
        return self.last_angle

    # ================= UPDATE =================
    def update(self, row, predictor=None):
        if row is None:
            return self.result()

        shoulder = _p(row, 12)
        elbow = _p(row, 14)
        wrist = _p(row, 16)

        angle = self.smooth_angle(shoulder, elbow, wrist)

        self.angle_series.append(angle)
        self.rows.append(row)

        # ===== DIRECTION =====
        if self.prev_angle is not None:
            if angle < self.prev_angle:
                self.direction_buffer.append("UP")
            elif angle > self.prev_angle:
                self.direction_buffer.append("DOWN")

        self.prev_angle = angle

        if self.direction_buffer:
            self.direction = max(set(self.direction_buffer), key=self.direction_buffer.count)

        # ===== CALIBRATION =====
        if not self.calibrated:
            self.calibration.append(angle)

            if len(self.calibration) >= CALIBRATION_FRAMES:
                mn, mx = min(self.calibration), max(self.calibration)

                if mx - mn > 15:
                    self.min_angle = mn
                    self.max_angle = mx

                    self.up_trigger = mx - UP_TRIGGER_OFFSET
                    self.down_trigger = mx - DOWN_TRIGGER_OFFSET

                    print("CALIBRATED:", mn, mx)

                    self.calibrated = True

            return self.result()

        # ===== FEATURES =====
        features, _ = build_features(
            list(self.rows),
            30,
            angle_series=np.array(self.angle_series, dtype=np.float32),
        )

        self.sequence.append(features[-1])
        velocity = float(features[-1][1])

        # ===== STATE MACHINE =====
        if self.direction == "UP" and angle < self.up_trigger:
            if self.state != "UP":
                self.state = "UP"
                self.rep_frames = 0
                self.rep_angles = []
                self.rep_velocities = []

        elif self.direction == "DOWN" and angle > self.down_trigger:

            if self.state == "UP" and self.rep_frames > MIN_REP_FRAMES and self.cooldown == 0:

                self.rep_count += 1

                rom = max(self.rep_angles) - min(self.rep_angles)
                full_range = max(self.max_angle - self.min_angle, 1e-6)
                ratio = rom / full_range

                avg_speed = np.mean(self.rep_velocities) if self.rep_velocities else 0

                # ===== FORM =====
                # ===== FINAL BALANCED FORM =====
                if ratio >= GOOD_RATIO:
                 self.form = "Correct"

                elif ratio <= BAD_RATIO:
                 self.form = "Incorrect"

                else:
                # TRUE borderline zone (no bias)
                 if self.last_prob > 0.55:
                  self.form = "Correct"
                 elif self.last_prob < 0.45:
                  self.form = "Incorrect"
                 else:
                  self.form = "Uncertain"

                # ===== SCORE =====
                score = self.score_rep(ratio, avg_speed)
                self.last_rep_score = score

                if score > 80:
                    self.last_rep_status = "EXCELLENT"
                elif score > 60:
                    self.last_rep_status = "GOOD"
                else:
                    self.last_rep_status = "BAD"

                self.cooldown = REP_COOLDOWN

            self.state = "DOWN"

        # cooldown
        if self.cooldown > 0:
            self.cooldown -= 1

        # ===== COLLECT =====
        if self.state == "UP":
            self.rep_frames += 1
            self.rep_angles.append(angle)
            self.rep_velocities.append(abs(velocity))

        # ===== MODEL (optional) =====
        if predictor and len(self.sequence) == SEQUENCE_LENGTH:
            _, prob = predictor.predict_probability(np.array(self.sequence, dtype=np.float32))
            self.last_prob = prob

        return self.result()

    # ================= SCORE =================
    def score_rep(self, ratio, speed):
        score = 0

        # ROM (relative)
        if ratio >= 0.75:
            score += 60
        elif ratio >= 0.6:
            score += 50
        elif ratio >= 0.45:
            score += 35
        else:
            score += 20

        # SPEED
        if speed < 0.05:
            score += 30
        elif speed < 0.08:
            score += 20
        else:
            score += 10

        return min(score, 100)

    # ================= OUTPUT =================
    def result(self):
        return {
            "state": self.state,
            "reps": self.rep_count,
            "form": self.form,
            "angle": self.last_angle or 0,
            "rep_score": self.last_rep_score,
            "rep_status": self.last_rep_status,
        }


# ================= CAMERA =================
def run_live_camera(model_path):
    predictor = ElbowInference(model_path)
    tracker = LiveRepTracker()

    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            row, res = extract_frame_landmarks(frame, pose)

            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            r = tracker.update(row, predictor)
            _draw(frame, r)

            cv2.imshow("Elbow AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def _draw(frame, r):
    if r["form"] == "Correct":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.putText(frame, f"Reps: {r['reps']}", (20, 40), 0, 0.8, color, 2)
    cv2.putText(frame, f"Form: {r['form']}", (20, 70), 0, 0.7, color, 2)
    cv2.putText(frame, f"Angle: {r['angle']:.1f}", (20, 100), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"State: {r['state']}", (20, 130), 0, 0.7, (255,255,255), 2)

    if r["rep_status"] != "WAITING":
        cv2.putText(frame, f"{r['rep_status']} ({r['rep_score']})", (20, 200), 0, 0.8, (0,255,255), 2)


def _p(row, i):
    return np.array([
        row.get(f"landmark_{i}_x",0),
        row.get(f"landmark_{i}_y",0),
        row.get(f"landmark_{i}_z",0),
    ])


if __name__ == "__main__":
    run_live_camera(Path(__file__).resolve().parents[2] / "models" / "elbow_flexion_lstm.pth")
