"""
ResNet-18 regression head for satellite nightlight tile analysis.

Predicts a continuous economic activity proxy (log GDP per capita or
informal-sector share) from VIIRS annual nightlight composite tiles.

References
----------
Chen & Nordhaus (2011) Using luminosity data as a proxy for economic statistics.
https://doi.org/10.1073/pnas.1017031108

Henderson, Storeygard & Weil (2012) Measuring economic growth from outer space.
https://doi.org/10.1257/aer.102.2.994
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class NightlightRegressor(nn.Module):
    """
    ResNet-18 backbone (ImageNet pre-trained) with a custom regression head.

    Input : (B, C, H, W)  — satellite tile, C=1 (single-band luminosity)
                             or C=3 (RGB composite for multi-band variants)
    Output: (B,)           — predicted log per-capita economic activity
    """

    def __init__(
        self,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt first conv to single-band input while keeping ImageNet weights
        if in_channels != 3:
            orig = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )
            if pretrained:
                # Average the 3 input channels
                backbone.conv1.weight.data = (
                    orig.weight.data.mean(dim=1, keepdim=True)
                )

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()   # remove classification head
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # (B, 512)
        return self.head(feats).squeeze(-1)


class TileDatasetInfo:
    """
    Metadata constants for VIIRS annual composite tiles.
    Tile size and normalisation statistics derived from the global
    2020 VNL v2 annual composite (DNB, cloud-free).
    """
    TILE_SIZE   = 256          # pixels
    PIXEL_SCALE = 0.00449      # degrees per pixel (~500 m at equator)
    # Per-channel stats (log1p of radiance, nW/cm²/sr)
    MEAN = 0.847
    STD  = 1.203
