"""Evaluation metrics for the elbow form classifier."""

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.features.elbow_features import INPUT_SIZE
from src.models.elbow_lstm import ElbowLSTM
from src.training.dataset import ElbowDataset


def evaluate(processed_dir: str | Path, model_path: str | Path) -> dict:
    dataset = ElbowDataset(processed_dir)
    loader = DataLoader(dataset, batch_size=8)
    model = ElbowLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for features, batch_labels in loader:
            predictions.extend(torch.argmax(model(features), dim=1).tolist())
            labels.extend(batch_labels.tolist())
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(evaluate(root / "processed", root / "models" / "elbow_flexion_lstm.pth"))
