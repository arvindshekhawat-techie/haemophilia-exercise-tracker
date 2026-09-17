"""Build fixed-length processed repetitions from labeled videos."""

from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

from src.features.elbow_features import (
    INPUT_SIZE,
    build_features,
    elbow_angle,
    filter_angle_outliers,
    smooth_angles,
)
from src.pose.pose_extraction import extract_video_landmarks, video_fps
from src.training.dataset import SEQUENCE_LENGTH

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ================= MAIN ================= #

def preprocess_dataset(dataset_dir: str | Path, processed_dir: str | Path) -> int:
    dataset_dir = Path(dataset_dir)
    processed_dir = Path(processed_dir)

    # 🔥 Clean previous
    if processed_dir.exists():
        for f in processed_dir.rglob("*.npy"):
            f.unlink()

    saved = {"correct": 0, "incorrect": 0}
    video_counts = {"correct": 0, "incorrect": 0}

    for source_class in ("correct", "incorrect"):
        class_dir = dataset_dir / source_class
        if not class_dir.exists():
            continue

        for video_path in sorted(class_dir.iterdir()):
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            label = 1 if source_class == "correct" else 0
            class_name = source_class

            output_dir = processed_dir / class_name
            output_dir.mkdir(parents=True, exist_ok=True)

            video_counts[class_name] += 1

            rows = extract_video_landmarks(video_path)
            if len(rows) == 0:
                print(f"⚠️ No landmarks: {video_path.name}")
                continue

            # -------- ANGLES -------- #
            raw_angles = np.array([elbow_angle(r) for r in rows], dtype=np.float32)
            angles = smooth_angles(filter_angle_outliers(raw_angles), window=7)

            fps = video_fps(video_path)

            # -------- PEAK DETECTION (KEY FIX) -------- #
            peaks, _ = find_peaks(
                -angles,                   # minima = bent
                distance=int(fps * 0.5),  # min gap between reps
                prominence=5              # ignore noise
            )

            boundaries = []

            for i in range(len(peaks) - 1):
                start = peaks[i]
                end = peaks[i + 1]

                if end - start < int(fps * 0.3):  # too short → skip noise
                    continue

                boundaries.append((start, end))

            # 🔥 Fallback ONLY if nothing detected
            if not boundaries:
                print(f"⚠️ No reps detected → fallback FULL video: {video_path.name}")
                boundaries = [(0, len(angles) - 1)]

            print(f"{class_name}/{video_path.name} → reps: {len(boundaries)}")

            # -------- SAVE REPS -------- #
            for i, (start, end) in enumerate(boundaries):
                rep_rows = rows[start:end + 1]
                rep_angles = angles[start:end + 1]

                features, _ = build_features(rep_rows, fps, angle_series=rep_angles)
                sequence = _resize_sequence(features)

                save_path = output_dir / f"{video_path.stem}_rep{i}.npy"

                np.save(
                    save_path,
                    {"features": sequence, "label": label},
                    allow_pickle=True
                )

                saved[class_name] += 1

                if i < 2:
                    print(
                        f"   Rep {i+1}: {rep_angles[0]:.0f} → {rep_angles.min():.0f} → {rep_angles[-1]:.0f}"
                    )

    # -------- SUMMARY -------- #
    total = saved["correct"] + saved["incorrect"]

    print("\n===== DATASET SUMMARY =====")
    print(f"Videos: {sum(video_counts.values())}")
    print(f"Correct videos: {video_counts['correct']}")
    print(f"Incorrect videos: {video_counts['incorrect']}")
    print(f"Total reps: {total}")
    print(f"Correct reps: {saved['correct']}")
    print(f"Incorrect reps: {saved['incorrect']}")

    if total > 0:
        imbalance = abs(saved["correct"] - saved["incorrect"]) / total
        print(f"Imbalance ratio: {imbalance:.2f}")

    return total


# ================= RESIZE ================= #

def _resize_sequence(features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros((SEQUENCE_LENGTH, INPUT_SIZE), dtype=np.float32)

    if len(features) < SEQUENCE_LENGTH:
        pad = np.repeat(features[-1:], SEQUENCE_LENGTH - len(features), axis=0)
        return np.vstack((features, pad)).astype(np.float32)

    if len(features) > SEQUENCE_LENGTH:
        idx = np.linspace(0, len(features) - 1, SEQUENCE_LENGTH)
        base = np.arange(len(features))

        return np.column_stack([
            np.interp(idx, base, features[:, i])
            for i in range(features.shape[1])
        ]).astype(np.float32)

    return features.astype(np.float32)


# ================= RUN ================= #

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    count = preprocess_dataset(
        root / "dataset" / "elbow_flexion",
        root / "processed"
    )

    print(f"\n✅ Saved {count} processed repetitions")