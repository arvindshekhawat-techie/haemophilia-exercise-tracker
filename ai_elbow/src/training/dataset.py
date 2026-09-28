from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch

SEQUENCE_LENGTH = 128
INPUT_SIZE = 8   # ✅ FIXED


class ElbowDataset(Dataset):
    def __init__(self, processed_dir):
        self.files = sorted(Path(processed_dir).glob("**/*.npy"))

        if not self.files:
            raise ValueError("No processed data found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = np.load(self.files[idx], allow_pickle=True).item()

        x = np.array(item["features"], dtype=np.float32)
        y = int(item["label"])

        # 🔥 SAFETY CHECK (VERY IMPORTANT)
        if x.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
            raise ValueError(f"Bad shape {x.shape} in {self.files[idx]}")

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )