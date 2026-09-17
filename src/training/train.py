"""Train and validate the elbow form classifier on repetition samples."""

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.features.elbow_features import INPUT_SIZE
from src.models.elbow_lstm import ElbowLSTM
from src.training.dataset import ElbowDataset


FEATURE_NAMES = ("normalized_angle", "velocity", "smoothness", "sin_angle", "cos_angle", "rom")


# ================= MAIN TRAIN ================= #

def train(processed_dir, model_path, epochs=40, batch_size=8, patience=12):
    dataset = ElbowDataset(processed_dir)
    _verify_samples(dataset.files)

    labels = np.array([_read_label(p) for p in dataset.files])

    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Both correct and incorrect reps required")

    print("\nTotal samples:", len(labels))
    _print_distribution("All", labels)

    train_idx, val_idx = _split_indices(dataset.files, labels)

    train_loader = DataLoader(
        Subset(ElbowDataset(processed_dir, files=dataset.files, augment=True), train_idx),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        Subset(ElbowDataset(processed_dir, files=dataset.files, augment=False), val_idx),
        batch_size=batch_size,
    )

    model = ElbowLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2)

    # ✅ STABLE OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=1e-4)

    # ✅ BETTER CLASS BALANCE
    class_weights = torch.tensor([1.0, 1.6])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        _print_unique_predictions(model, train_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            _save_model(model, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    print("\n=== FINAL VALIDATION ===")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    metrics = _metrics(model, val_loader)

    for k, v in metrics.items():
        print(f"{k}: {v}")


# ================= TRAIN LOOP ================= #

def _train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0

    for x, y in loader:
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        preds = _predict(logits)

        total_loss += loss.item() * len(y)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def _eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)

            preds = _predict(logits)

            total_loss += loss.item() * len(y)
            correct += (preds == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


# ================= SMART PREDICTION ================= #

def _predict(logits):
    probs = torch.softmax(logits, dim=1)[:, 1]

    # ✅ Better decision boundary (prevents collapse)
    preds = torch.zeros_like(probs, dtype=torch.long)

    preds[probs > 0.6] = 1      # confident correct
    preds[probs < 0.4] = 0      # confident incorrect

    # between 0.4–0.6 → keep neutral → push to incorrect
    preds[(probs >= 0.4) & (probs <= 0.6)] = 0

    return preds


# ================= METRICS ================= #

def _metrics(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            p = _predict(model(x))
            preds.extend(p.tolist())
            labels.extend(y.tolist())

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


# ================= DATA SPLIT ================= #

def _split_indices(files, labels):
    groups = np.array([p.name.split("_rep")[0] for p in files])
    unique, idx = np.unique(groups, return_inverse=True)

    group_labels = np.array([labels[np.where(idx == i)[0][0]] for i in range(len(unique))])

    train_g, val_g = train_test_split(
        np.arange(len(unique)),
        test_size=0.2,
        stratify=group_labels,
        random_state=42,
    )

    train = [i for i, g in enumerate(idx) if g in train_g]
    val = [i for i, g in enumerate(idx) if g in val_g]

    return train, val


# ================= DEBUG ================= #

def _verify_samples(files):
    if not files:
        raise ValueError("No data found")

    x = np.concatenate([np.load(f, allow_pickle=True).item()["features"] for f in files])
    print("Shape:", x.shape)

    for i, name in enumerate(FEATURE_NAMES):
        print(name, x[:, i].min(), x[:, i].max())


def _print_distribution(name, labels):
    print(f"{name}: incorrect={sum(labels==0)}, correct={sum(labels==1)}")


def _print_unique_predictions(model, loader):
    model.eval()
    preds = []

    with torch.no_grad():
        for x, _ in loader:
            preds.append(_predict(model(x)))

    print("Unique preds:", torch.unique(torch.cat(preds)))


def _save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _read_label(p):
    return int(np.load(p, allow_pickle=True).item()["label"])


# ================= MAIN ================= #

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train(root / "processed", root / "models" / "elbow_flexion_lstm.pth")