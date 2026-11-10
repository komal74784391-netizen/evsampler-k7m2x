import torch.nn as nn


class SparseDepthNet(nn.Module):
    """Sparse-to-dense depth completion network."""

    def __init__(self, in_channels=2, base_width=32):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, base_width, 3, padding=1),
            nn.Conv2d(base_width, base_width * 2, 3, stride=2, padding=1),
            nn.Conv2d(base_width * 2, base_width * 4, 3, stride=2, padding=1),
        ])
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(base_width * 4, base_width * 2, 2, stride=2),
            nn.ConvTranspose2d(base_width * 2, base_width, 2, stride=2),
            nn.Conv2d(base_width, 1, 1),
        ])

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        for layer in self.decoder:
            x = layer(x)
        return x
