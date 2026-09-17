"""Real-time elbow inference — FINAL (ANGLE-DRIVEN FORM, NO BIAS)"""

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
ANGLE_DEADZONE = 1.2

CALIBRATION_FRAMES = 40

MIN_REP_FRAMES = 12
REP_COOLDOWN = 8

PROB_SMOOTH = 10
DIRECTION_SMOOTH = 4

# 🔥 KEY FIX: ROM thresholds (THIS FIXES YOUR PROBLEM)
GOOD_ROM = 75
BAD_ROM = 55
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
        self.flexion = None
        self.extension = None

        # Rep tracking
        self.rep_count = 0
        self.rep_frames = 0
        self.cooldown = 0

        self.rep_angles = []
        self.rep_velocities = []

        # Model
        self.prob_buffer = deque(maxlen=PROB_SMOOTH)
        self.last_prob = 0.5

        # Form
        self.form = "WAITING"
        self.form_locked = False
        self.frozen_form = "WAITING"

        # Output
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

                if mx - mn > 20:
                    self.flexion = mn + 0.4 * (mx - mn)
                    self.extension = mx - 0.25 * (mx - mn)
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
        if self.direction == "UP" and angle < self.flexion:
            self.state = "UP"
            self.form_locked = False

        elif self.direction == "DOWN" and angle > self.extension:
            if self.state == "UP" and self.rep_frames > MIN_REP_FRAMES and self.cooldown == 0:

                self.rep_count += 1

                # 🔥 FINAL FORM DECISION (ANGLE BASED)
                rom = max(self.rep_angles) - min(self.rep_angles)

                if rom >= GOOD_ROM:
                    self.form = "Correct"
                elif rom <= BAD_ROM:
                    self.form = "Incorrect"
                else:
                    # borderline → use model
                    if self.last_prob > 0.55:
                        self.form = "Correct"
                    else:
                        self.form = "Incorrect"

                # freeze result
                self.frozen_form = self.form
                self.form_locked = True

                # score
                score = self.score_rep(rom, velocity)
                self.last_rep_score = score

                if score > 80:
                    self.last_rep_status = "EXCELLENT"
                elif score > 55:
                    self.last_rep_status = "GOOD"
                else:
                    self.last_rep_status = "BAD"

                self.cooldown = REP_COOLDOWN

            self.state = "DOWN"

        # cooldown
        if self.cooldown > 0:
            self.cooldown -= 1

        # collect rep
        if self.state == "UP":
            self.rep_frames += 1
            self.rep_angles.append(angle)
            self.rep_velocities.append(abs(velocity))
        else:
            self.rep_frames = 0
            self.rep_angles = []
            self.rep_velocities = []

        # ===== MODEL (ONLY SUPPORT) =====
        if predictor and len(self.sequence) == SEQUENCE_LENGTH:
            _, prob = predictor.predict_probability(np.array(self.sequence, dtype=np.float32))
            self.prob_buffer.append(prob)
            self.last_prob = float(np.mean(self.prob_buffer))

        if self.form_locked:
            self.form = self.frozen_form

        return self.result()

    # ================= SCORE =================
    def score_rep(self, rom, velocity):
        score = 0

        # ROM (main)
        if rom > 100:
            score += 60
        elif rom > 80:
            score += 45
        elif rom > 60:
            score += 25
        else:
            score += 10

        # SPEED
        if velocity < 0.03:
            score += 30
        elif velocity < 0.06:
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
            "prob": self.last_prob,
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
    elif r["form"] == "Incorrect":
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)

    cv2.putText(frame, f"Reps: {r['reps']}", (20, 40), 0, 0.8, color, 2)
    cv2.putText(frame, f"Form: {r['form']}", (20, 70), 0, 0.7, color, 2)
    cv2.putText(frame, f"Angle: {r['angle']:.1f}", (20, 100), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"State: {r['state']}", (20, 130), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Prob: {r['prob']:.2f}", (20, 160), 0, 0.6, (255,255,255), 2)

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