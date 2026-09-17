"""Temporal features for elbow repetitions, including range of motion."""

from typing import Dict, Sequence, Tuple

import numpy as np


INPUT_SIZE = 6
SHOULDER = 12
ELBOW = 14
WRIST = 16
VELOCITY_WINDOW = 5


def elbow_angle(landmarks: Dict[str, float]) -> float:
    """Compute a camera-robust joint angle from relative 3D vectors."""
    shoulder = _point(landmarks, SHOULDER)
    elbow = _point(landmarks, ELBOW)
    wrist = _point(landmarks, WRIST)
    first = shoulder - elbow
    second = wrist - elbow
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-6:
        return 0.0
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def build_features(
    landmark_rows: Sequence[Dict[str, float]],
    fps: float = 30.0,
    angle_series: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the fixed six-feature contract for a frame sequence."""
    if not landmark_rows:
        return np.empty((0, INPUT_SIZE), dtype=np.float32), np.empty(0, dtype=np.float32)
    angles = np.asarray(
        angle_series if angle_series is not None else [elbow_angle(row) for row in landmark_rows],
        dtype=np.float32,
    )
    angles = smooth_angles(angles, window=5)
    velocity = np.zeros_like(angles)
    if len(angles) > 1:
        velocity[1:] = angles[1:] - angles[:-1]
    velocity = np.clip(velocity / 50.0, -1.0, 1.0)
    smoothness = np.clip(np.abs(np.diff(angles, prepend=angles[0])) / 50.0, 0.0, 1.0)
    normalized_angle = angles / 180.0
    radians = np.radians(angles)
    rom = (np.max(angles) - np.min(angles)) / 180.0
    rom_feature = np.full_like(angles, rom)
    features = np.stack(
        (
            normalized_angle,
            velocity,
            smoothness,
            np.sin(radians),
            np.cos(radians),
            rom_feature,
        ),
        axis=1,
    )
    return features.astype(np.float32), angles


def smooth_angles(angles: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a centered moving average without changing sequence length."""
    angles = np.asarray(angles, dtype=np.float32)
    if len(angles) < 2 or window <= 1:
        return angles
    window = min(window, len(angles) if len(angles) % 2 else len(angles) - 1)
    if window < 2:
        return angles
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(angles, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def filter_angle_outliers(angles: np.ndarray, max_jump: float = 30.0) -> np.ndarray:
    """Replace isolated angle spikes with the local median."""
    angles = np.asarray(angles, dtype=np.float32).copy()
    if len(angles) < 3:
        return angles
    median = smooth_angles(angles, window=3)
    spikes = np.abs(angles - median) > max_jump
    angles[spikes] = median[spikes]
    return angles


def _point(row: Dict[str, float], index: int) -> np.ndarray:
    return np.asarray(
        [row.get(f"landmark_{index}_{axis}", 0.0) for axis in ("x", "y", "z")],
        dtype=np.float32,
    )
