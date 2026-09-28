from pathlib import Path
import numpy as np
import shutil
from collections import defaultdict
import random

# =========================
# LOAD ORIGINAL FILES
# =========================
source_path = Path("processed")
files = list(source_path.rglob("*.npy"))

print("Total files:", len(files))

# =========================
# CREATE CLEAN FOLDER
# =========================
clean_path = Path("processed_clean")

if clean_path.exists():
    shutil.rmtree(clean_path)

clean_path.mkdir()

# =========================
# STEP 1: ANALYZE RANGES
# =========================
ranges = []

for f in files:
    try:
        data = np.load(f, allow_pickle=True).item()
        angles = data["features"][:, 0]

        r = float(np.max(angles) - np.min(angles))
        ranges.append(r)

    except:
        continue

# 🔥 DEBUG RANGE DISTRIBUTION
print("\nRange stats:")
print(f"Min: {min(ranges):.3f}")
print(f"Max: {max(ranges):.3f}")
print(f"Mean: {np.mean(ranges):.3f}")

# =========================
# STEP 2: SMART FILTER
# =========================
kept_files = []

LOW = max(0.08, np.percentile(ranges, 10))   # adaptive lower bound
HIGH = min(1.2, np.percentile(ranges, 95))   # adaptive upper bound

print(f"\nUsing thresholds → LOW: {LOW:.3f}, HIGH: {HIGH:.3f}")

for f in files:
    try:
        data = np.load(f, allow_pickle=True).item()
        angles = data["features"][:, 0]

        r = float(np.max(angles) - np.min(angles))

        # 🔥 FINAL FILTER
        if LOW < r < HIGH:
            kept_files.append(f)

    except:
        print("Skipping bad file:", f)

print("\nAfter filtering:")
print("Files kept:", len(kept_files))

# =========================
# STEP 3: GROUP BY CLASS
# =========================
class_files = defaultdict(list)

for f in kept_files:
    data = np.load(f, allow_pickle=True).item()
    label = data["label"]
    class_files[label].append(f)

count_0 = len(class_files[0])
count_1 = len(class_files[1])

print("\nBefore balancing:")
print("Class 0:", count_0)
print("Class 1:", count_1)

if count_0 == 0 or count_1 == 0:
    print("\n❌ ERROR: One class empty!")
    print("👉 Adjust thresholds or check preprocessing")
    exit()

# =========================
# STEP 4: BALANCE DATASET
# =========================
min_count = min(count_0, count_1)

random.seed(42)

balanced_files = []
balanced_files += random.sample(class_files[0], min_count)
balanced_files += random.sample(class_files[1], min_count)

print("\nAfter balancing:")
print("Each class:", min_count)

# =========================
# STEP 5: COPY FILES
# =========================
for f in balanced_files:
    dst = clean_path / f.name

    if f.exists():
        shutil.copy(f, dst)
    else:
        print("Missing file skipped:", f)

print("\n✅ Final dataset ready!")
print("Total:", len(balanced_files))