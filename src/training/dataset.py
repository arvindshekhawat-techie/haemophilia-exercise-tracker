"""Dataset for fixed-length processed elbow repetitions."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.elbow_features import INPUT_SIZE

SEQUENCE_LENGTH = 128


class ElbowDataset(Dataset):
    def __init__(self, processed_dir: str | Path, files: List[Path] | None = None, augment: bool = False):
        self.processed_dir = Path(processed_dir)
        self.files = files or sorted(self.processed_dir.glob("**/*.npy"))
        self.augment = augment
        if not self.files:
            raise ValueError(f"No processed repetitions found in {self.processed_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = np.load(self.files[index], allow_pickle=True).item()
        features = np.asarray(item["features"], dtype=np.float32)
        if features.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
            raise ValueError(
                f"Expected {(SEQUENCE_LENGTH, INPUT_SIZE)}, got {features.shape} in {self.files[index]}"
            )
        if self.augment:
            features = _augment_sequence(features)
        return torch.from_numpy(features), torch.tensor(int(item["label"]), dtype=torch.long)


def _augment_sequence(features: np.ndarray) -> np.ndarray:
    """Add only the requested small, feature-preserving training noise."""
    return (features + np.random.normal(0.0, 0.005, features.shape)).astype(np.float32)
