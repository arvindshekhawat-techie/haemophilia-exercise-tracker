"""Adaptive elbow repetition detection and rep-level metrics."""

from typing import Dict, List

import numpy as np


EXTENDED = "EXTENDED"
FLEXED = "FLEXED"


def adaptive_thresholds(angles: np.ndarray) -> tuple[float, float, float, float, float]:
    if len(angles) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    minimum = float(np.min(angles))
    maximum = float(np.max(angles))
    movement = maximum - minimum
    return minimum, maximum, movement, minimum + 0.4 * movement, maximum - 0.4 * movement


def detect_rep_boundaries(angles: np.ndarray, min_frames: int = 10) -> List[tuple[int, int]]:
    """Detect complete extended-to-flexed-to-extended repetitions."""
    _, _, movement, flex_threshold, extend_threshold = adaptive_thresholds(angles)
    if movement < 15:
        return []
    state = EXTENDED
    start = None
    boundaries = []
    for index, angle in enumerate(angles):
        if state == EXTENDED and angle > extend_threshold:
            start = index if start is None else start
        elif state == EXTENDED and angle < flex_threshold and start is not None:
            state = FLEXED
        elif state == FLEXED and angle > extend_threshold and start is not None:
            if index - start >= min_frames:
                boundaries.append((start, index))
            state = EXTENDED
            start = index
    return boundaries


def detect_reps(angles: np.ndarray, features: np.ndarray, fps: float = 30.0) -> List[Dict]:
    if len(angles) != len(features):
        raise ValueError("Angles and features must contain the same number of frames")
    reps = []
    for start, end in detect_rep_boundaries(angles):
        segment = slice(start, end + 1)
        segment_angles = angles[segment]
        segment_features = features[segment]
        duration = (end - start) / max(fps, 1.0)
        reps.append(
            {
                "features": segment_features,
                "angles": segment_angles,
                "start_frame": start,
                "end_frame": end,
                "rom": float(np.ptp(segment_angles)),
                "duration": float(duration),
                "speed": float(np.ptp(segment_angles) / duration) if duration else 0.0,
                "smoothness": float(np.mean(segment_features[:, 2])),
                "avg_velocity": float(np.mean(np.abs(segment_features[:, 1] * 180.0))),
                "quality": float(np.mean(segment_features[:, 2])),
            }
        )
    return reps


def rep_score(metrics: Dict, form: str) -> float:
    """Score a completed repetition from its range of motion."""
    rom = float(metrics.get("rom", 0.0))
    if rom > 100:
        return 100.0
    if rom > 70:
        return 70.0
    return 0.0
