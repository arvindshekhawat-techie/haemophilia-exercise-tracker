from pathlib import Path
import numpy as np

from src.pose.pose_extraction import extract_video_landmarks, video_fps
from src.features.elbow_features import build_features
from src.training.dataset import SEQUENCE_LENGTH, INPUT_SIZE


# ==============================
# RESIZE TO FIXED LENGTH
# ==============================
def _resize_sequence(features):
    if len(features) == 0:
        return np.zeros((SEQUENCE_LENGTH, INPUT_SIZE), dtype=np.float32)

    if len(features) < SEQUENCE_LENGTH:
        pad = np.repeat(features[-1:], SEQUENCE_LENGTH - len(features), axis=0)
        return np.vstack((features, pad)).astype(np.float32)

    if len(features) > SEQUENCE_LENGTH:
        idx = np.linspace(0, len(features) - 1, SEQUENCE_LENGTH).astype(int)
        return features[idx].astype(np.float32)

    return features.astype(np.float32)


# ==============================
# ANGLE FUNCTION
# ==============================
def elbow_angle(row):
    a = np.array([row[f"landmark_12_{x}"] for x in ("x", "y", "z")])
    b = np.array([row[f"landmark_14_{x}"] for x in ("x", "y", "z")])
    c = np.array([row[f"landmark_16_{x}"] for x in ("x", "y", "z")])

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

    return float(np.clip(angle, 40, 170))


# ==============================
# 🔥 NEW ROBUST REP DETECTOR
# ==============================
def detect_reps(angles_smooth):
    boundaries = []

    start = 0
    n = len(angles_smooth)

    while start < n - 20:
        # find local min (flexed)
        search_window = angles_smooth[start: min(start + 120, n)]
        if len(search_window) < 10:
            break

        min_idx = start + np.argmin(search_window)

        # find next max after min
        search_end = min(min_idx + 120, n)
        max_idx = min_idx + np.argmax(angles_smooth[min_idx:search_end])

        start_angle = angles_smooth[start]
        mid_angle = angles_smooth[min_idx]
        end_angle = angles_smooth[max_idx]

        # 🔥 VALIDATION (tune here if needed)
        if (
            start_angle > 120 and
            mid_angle < 120 and
            end_angle > 120 and
            (start_angle - mid_angle) > 20 and
            (end_angle - mid_angle) > 20 and
            (max_idx - start) > 10   # avoid tiny reps
        ):
            boundaries.append((start, max_idx, min_idx))

            # jump forward → avoid duplicate detection
            start = max_idx
        else:
            start += 10  # slide window

    return boundaries


# ==============================
# MAIN PREPROCESS (FIXED)
# ==============================
def preprocess_dataset(dataset_dir, processed_dir):
    dataset_dir = Path(dataset_dir)
    processed_dir = Path(processed_dir)

    if processed_dir.exists():
        for f in processed_dir.rglob("*.npy"):
            f.unlink()

    for class_name in ["correct", "incorrect"]:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue

        label = 1 if class_name == "correct" else 0

        save_dir = processed_dir / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for video_path in class_dir.glob("*.mp4"):
            print(f"\nProcessing: {video_path.name}")

            rows = extract_video_landmarks(video_path)
            angles = np.array([elbow_angle(r) for r in rows], dtype=np.float32)

            # smooth
            angles_smooth = np.convolve(angles, np.ones(5)/5, mode='same')

            fps = video_fps(video_path)

            # -----------------------------
            # 🔥 NEW RANGE-BASED REP DETECTION
            # -----------------------------
            boundaries = []
            start = 0
            n = len(angles_smooth)

            while start < n - 20:
                window = angles_smooth[start:start+150]

                if len(window) < 20:
                    break

                min_idx = start + np.argmin(window)
                max_idx = min_idx + np.argmax(angles_smooth[min_idx:min(min_idx+150, n)])

                start_angle = angles_smooth[start]
                mid_angle = angles_smooth[min_idx]
                end_angle = angles_smooth[max_idx]

                range_motion = start_angle - mid_angle

                # -----------------------------
                # 🔥 QUALITY FILTER (NOT LABEL BASED)
                # -----------------------------
                if (
                    start_angle > 120 and
                    mid_angle < 130 and
                    end_angle > 120 and
                    range_motion > 15 and
                    (max_idx - start) > int(fps * 0.5)
                ):
                    boundaries.append((start, max_idx, min_idx))
                    start = max_idx
                else:
                    start += 10

            print(f"Detected reps: {len(boundaries)}")

            # -----------------------------
            # SAVE REPS
            # -----------------------------
            for i, (start, end, mid) in enumerate(boundaries):
                rep_rows = rows[start:end]
                rep_angles = angles_smooth[start:end]

                # 🔥 EXTRA METADATA FOR MODEL LEARNING
                range_motion = np.max(rep_angles) - np.min(rep_angles)
                duration = len(rep_angles) / fps
                peak = np.max(rep_angles)

                # 🚨 IMPORTANT: mild filtering (NOT aggressive)
                if range_motion < 10:
                    continue

                features, _ = build_features(rep_rows, fps, rep_angles)
                sequence = _resize_sequence(features)

                save_path = save_dir / f"{video_path.stem}_rep{i}.npy"

                np.save(save_path, {
                    "features": sequence,
                    "label": label,
                    "range": range_motion,
                    "duration": duration,
                    "peak": peak
                })

                print(f"Saved rep {i+1}")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    preprocess_dataset(
        root / "dataset" / "elbow_flexion",
        root / "processed"
    )