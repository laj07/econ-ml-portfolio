"""
LSTM and Temporal Fusion Transformer for GDP nowcasting.

Given monthly high-frequency alternative data (Google Trends, nighttime lights,
mobile-mobility proxies), predict quarter-on-quarter real GDP growth for a
panel of countries — before official statistics are released.

References
----------
Lim et al. (2021) Temporal Fusion Transformers for interpretable
multi-horizon time series forecasting. IJF.
https://doi.org/10.1016/j.ijforecast.2021.03.012
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GDPNowcastLSTM(nn.Module):
    """
    Bidirectional LSTM baseline.

    Input : (batch, seq_len, n_features)  — monthly alt-data features
    Output: (batch,)                       — predicted QoQ GDP growth (%)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = self.norm(out[:, -1, :])
        return self.head(h).squeeze(-1)


class VariableSelectionNetwork(nn.Module):
    """
    Soft input-variable selector from the TFT paper.
    Returns a weighted combination of per-feature embeddings plus the
    selection weights (useful for feature-importance analysis).
    """

    def __init__(self, n_features: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_size), nn.ELU())
            for _ in range(n_features)
        ])
        self.selector = nn.Sequential(
            nn.Linear(n_features * hidden_size, n_features),
            nn.Softmax(dim=-1),
        )
        self.n = n_features
        self.h = hidden_size

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, f = x.shape
        transformed = torch.stack(
            [self.nets[i](x[:, :, i : i + 1]) for i in range(f)], dim=2
        )  # (b, t, f, h)
        flat    = transformed.reshape(b, t, f * self.h)
        weights = self.selector(flat)                          # (b, t, f)
        selected = (weights.unsqueeze(-1) * transformed).sum(dim=2)  # (b, t, h)
        return selected, weights


class NowcastTFT:
    """
    Thin factory wrapper around pytorch_forecasting.TemporalFusionTransformer.

    Use NowcastTFT.from_dataset(dataset) to construct a ready-to-train TFT
    with sensible defaults for macro nowcasting panels.
    """

    DEFAULTS: dict = dict(
        hidden_size=64,
        lstm_layers=2,
        dropout=0.1,
        output_size=7,           # 7 quantiles
        attention_head_size=4,
        max_encoder_length=24,   # 24 months of history
        max_prediction_length=4, # predict up to 4 quarters ahead
    )

    @staticmethod
    def from_dataset(dataset, **kwargs):
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_forecasting.metrics import QuantileLoss
        except ImportError as exc:
            raise ImportError(
                "pytorch-forecasting is required. "
                "pip install pytorch-forecasting"
            ) from exc

        params = {**NowcastTFT.DEFAULTS, **kwargs}
        return TemporalFusionTransformer.from_dataset(
            dataset,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
            **params,
        )
