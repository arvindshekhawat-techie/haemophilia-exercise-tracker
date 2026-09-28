import numpy as np

INPUT_SIZE = 8


def resample(seq, target_len=128):
    idx = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return seq[idx]


def build_features(rows, fps, angle_series):
    angle_series = np.array(angle_series, dtype=np.float32)

    if len(angle_series) < 5:
        return np.zeros((128, INPUT_SIZE), dtype=np.float32), angle_series

    # -------------------------
    # 🔥 RAW SIGNAL
    # -------------------------
    raw_angle = angle_series.copy()

    # -------------------------
    # 🔥 NORMALIZED SHAPE
    # -------------------------
    min_a = np.min(angle_series)
    max_a = np.max(angle_series)
    norm_angle = (angle_series - min_a) / (max_a - min_a + 1e-6)

    # -------------------------
    # 🔥 DERIVATIVES
    # -------------------------
    vel = np.diff(raw_angle, prepend=raw_angle[0])
    acc = np.diff(vel, prepend=vel[0])

    # -------------------------
    # 🔥 SHAPE FEATURES (VERY IMPORTANT)
    # -------------------------
    peak = np.max(raw_angle)
    valley = np.min(raw_angle)

    range_motion = peak - valley
    duration = len(raw_angle) / fps

    # 🔥 symmetry (correct reps are smoother)
    half = len(raw_angle) // 2
    first = raw_angle[:half]
    second = raw_angle[-half:][::-1]

    if len(first) == len(second) and len(first) > 0:
        symmetry = np.mean(np.abs(first - second))
    else:
        symmetry = 0.0

    # -------------------------
    # 🔥 RESAMPLE
    # -------------------------
    raw_angle = resample(raw_angle)
    norm_angle = resample(norm_angle)
    vel = resample(vel)
    acc = resample(acc)

    # -------------------------
    # 🔥 GLOBAL FEATURES (broadcast)
    # -------------------------
    range_norm = np.full(128, range_motion / 100)
    duration_norm = np.full(128, duration / 5)
    symmetry_norm = np.full(128, symmetry / 50)
    peak_norm = np.full(128, peak / 180)

    # -------------------------
    # 🔥 FINAL FEATURES (8 STRONG SIGNALS)
    # -------------------------
    features = np.column_stack([
        raw_angle / 180,     # 🔥 absolute info
        norm_angle,          # shape
        vel / 50,            # motion speed
        acc / 50,            # smoothness
        range_norm,          # full ROM
        duration_norm,       # control
        symmetry_norm,       # smooth rep
        peak_norm            # extension quality
    ])

    return features.astype(np.float32), angle_series