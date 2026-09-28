from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 🔥 FIXED: recursive search
files = list(Path("processed").glob("**/*.npy"))

print("Total files found:", len(files))

if len(files) == 0:
    print("❌ No data found. Check your processed folder path.")
    exit()

# =========================
# ✅ STEP 1 — LABEL COUNT
# =========================
count_0 = 0
count_1 = 0

for f in files:
    data = np.load(f, allow_pickle=True).item()
    y = data["label"]

    if y == 0:
        count_0 += 1
    else:
        count_1 += 1

print("Class 0 (incorrect):", count_0)
print("Class 1 (correct):", count_1)

# =========================
# ✅ STEP 2 — VISUAL CHECK
# =========================
correct = []
incorrect = []

for f in files:
    data = np.load(f, allow_pickle=True).item()
    angles = data["features"][:, 0]
    label = data["label"]

    if label == 1:
        correct.append(angles)
    else:
        incorrect.append(angles)

plt.figure(figsize=(10, 5))

# Green = correct
for a in correct[:5]:
    plt.plot(a, color='green', alpha=0.7)

# Red = incorrect
for a in incorrect[:5]:
    plt.plot(a, color='red', alpha=0.7)

plt.title("Green = Correct | Red = Incorrect")
plt.xlabel("Frame")
plt.ylabel("Angle")
plt.show()