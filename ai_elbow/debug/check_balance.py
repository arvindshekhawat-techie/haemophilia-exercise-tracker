from pathlib import Path
import numpy as np
from collections import Counter

labels = []

for f in Path("processed_clean").rglob("*.npy"):
    data = np.load(f, allow_pickle=True).item()
    labels.append(data["label"])

print("Class distribution:", Counter(labels))