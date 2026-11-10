import torch
import torch.nn as nn


class AdaptiveSamplingPolicy(nn.Module):
    """Predicts the per-pixel sampling probability from a short event window."""

    def __init__(self, in_channels=4, hidden=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, events):
        features = self.encoder(events)
        return torch.sigmoid(self.head(features)).squeeze(1)
