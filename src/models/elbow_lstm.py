"""LSTM classifier for elbow exercise form."""

import torch
from torch import nn


class ElbowLSTM(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(features)
        return self.classifier(output[:, -1, :])
