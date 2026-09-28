import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.training.dataset import ElbowDataset
from src.models.elbow_lstm import ElbowLSTM


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    dataset = ElbowDataset("processed_clean")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ElbowLSTM(input_size=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.3]).to(device)
    )

    for epoch in range(80):
        total_loss = 0
        correct = 0
        total = 0

        model.train()

        for x, y in loader:
            x = x.float().to(device)
            y = y.float().unsqueeze(1).to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            # 🔥 STABILITY BOOST
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

        print(f"Epoch {epoch+1} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/elbow_lstm.pth")


if __name__ == "__main__":
    train()