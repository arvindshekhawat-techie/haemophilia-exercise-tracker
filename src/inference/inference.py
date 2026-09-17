"""Rep-level offline inference with quality and prediction stability gates."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.exercises.elbow_flexion import detect_reps, rep_score
from src.features.elbow_features import INPUT_SIZE, build_features, elbow_angle, filter_angle_outliers, smooth_angles
from src.models.elbow_lstm import ElbowLSTM
from src.pose.pose_extraction import extract_video_landmarks, video_fps
from src.training.dataset import SEQUENCE_LENGTH


class ElbowInference:
    def __init__(self, model_path: str | Path, threshold: float = 0.55):
        self.model = ElbowLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.threshold = threshold

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probabilities = self._probabilities(features)
        positive_probability = float(probabilities[1])
        confidence = float(torch.max(probabilities))
        if confidence < self.threshold or abs(float(probabilities[1] - probabilities[0])) < 0.15:
            return "UNCERTAIN", confidence
        return ("CORRECT" if positive_probability > self.threshold else "INCORRECT", confidence)

    def predict_probability(self, features: np.ndarray) -> tuple[str, float]:
        """Return the model label and raw probability of the correct class."""
        probabilities = self._probabilities(features)
        positive_probability = float(probabilities[1])
        return ("CORRECT" if positive_probability > self.threshold else "INCORRECT", positive_probability)

    def _probabilities(self, features: np.ndarray) -> torch.Tensor:
        sequence = _resize_sequence(features)
        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(sequence[None])), dim=1)[0]

    def analyze_video(self, video_path: str | Path) -> dict:
        fps = video_fps(video_path)
        rows = extract_video_landmarks(video_path)
        if _low_confidence(rows):
            return {"form": "UNCERTAIN", "confidence": 0.0, "rep_count": 0, "score": 0.0, "error": "LOW_VISIBILITY"}
        raw_angles = np.asarray([elbow_angle(row) for row in rows], dtype=np.float32)
        angles = smooth_angles(filter_angle_outliers(raw_angles))
        features, angles = build_features(rows, fps, angle_series=angles)
        reps = detect_reps(angles, features, fps)
        predictions = [self.predict(rep["features"]) for rep in reps]
        labels = [prediction[0] for prediction in predictions]
        form = Counter(labels).most_common(1)[0][0] if labels else "UNCERTAIN"
        confidence = float(np.mean([prediction[1] for prediction in predictions])) if predictions else 0.0
        last = reps[-1] if reps else {}
        return {
            "form": form,
            "confidence": confidence,
            "rep_count": len(reps),
            "rom": last.get("rom", 0.0),
            "speed": last.get("speed", 0.0),
            "smoothness": last.get("smoothness", 0.0),
            "score": rep_score(last, form) if last else 0.0,
            "error": None if reps else "NO_COMPLETE_REP",
        }


def _resize_sequence(features: np.ndarray) -> np.ndarray:
    """Pad or resample time only; never normalize features within a rep."""
    if len(features) == 0:
        return np.zeros((SEQUENCE_LENGTH, INPUT_SIZE), dtype=np.float32)
    if len(features) < SEQUENCE_LENGTH:
        return np.vstack((features, np.repeat(features[-1:], SEQUENCE_LENGTH - len(features), axis=0))).astype(np.float32)
    positions = np.linspace(0, len(features) - 1, SEQUENCE_LENGTH)
    source = np.arange(len(features))
    return np.column_stack(
        [np.interp(positions, source, features[:, i]) for i in range(INPUT_SIZE)]
    ).astype(np.float32)


def _low_confidence(rows) -> bool:
    bad = sum(any(row.get(f"landmark_{i}_visibility", 0.0) < 0.5 for i in (12, 14, 16)) for row in rows)
    return bad / max(len(rows), 1) > 0.3
