"""
Experiment: GDP Nowcasting
============================
Trains an LSTM (baseline) and Temporal Fusion Transformer (main model) to
nowcast quarterly GDP growth for a panel of developing countries using
high-frequency alternative data.

Config keys (under exp_params):
    model:         "lstm" | "tft"
    hidden_size:   128
    num_layers:    2
    dropout:       0.2
    lr:            1e-3
    batch_size:    64
    max_epochs:    50
    countries:     ["IND","BRA","ZAF",...]
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.trainer import register
from src.utils.io import get_device, set_seed, save_checkpoint
from src.utils.metrics import mae, rmse, pearson_r

log = logging.getLogger(__name__)


def _make_sequences(
    df, target_col: str, feature_cols: list[str], seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over the time axis to create (X, y) pairs."""
    X_list, y_list = [], []
    for _, group in df.groupby("country"):
        group = group.sort_values("year")
        vals  = group[feature_cols].values.astype(np.float32)
        tgt   = group[target_col].values.astype(np.float32)
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i - seq_len : i])
            y_list.append(tgt[i])
    return np.stack(X_list), np.array(y_list, dtype=np.float32)


@register("gdp_nowcasting")
def run(cfg: dict) -> None:
    params    = cfg.get("exp_params", {})
    smoke     = params.get("smoke", False)
    device    = get_device(params.get("device"))
    set_seed(params.get("seed", 42))
    model_key = params.get("model", "lstm")

    log.info("Model: %s | Device: %s | Smoke: %s", model_key, device, smoke)

    # ── Data ─────────────────────────────────────────────────────────────────
    if smoke:
        from src.datasets.wdi import WDIPanel
        panel = WDIPanel.synthetic_panel(n_countries=4, n_years=24)
        log.info("Using synthetic panel for smoke test")
    else:
        from src.datasets.wdi import WDIPanel
        wdi   = WDIPanel(
            countries=params.get("countries", None),
            cache_dir=cfg.get("data", {}).get("wdi_dir", "data/wdi"),
        )
        panel = wdi.load()

    feature_cols = [c for c in panel.columns if c not in ("country", "year", "gdp_growth")]
    panel[feature_cols] = panel[feature_cols].fillna(panel[feature_cols].mean())
    panel["gdp_growth"]  = panel["gdp_growth"].fillna(0.0)

    seq_len = params.get("seq_len", 8)
    X, y    = _make_sequences(panel, "gdp_growth", feature_cols, seq_len)

    # Train / val split (80 / 20)
    n_train  = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    batch_size   = params.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_features = X.shape[2]

    if model_key == "lstm":
        from src.models.lstm import GDPNowcastLSTM
        model = GDPNowcastLSTM(
            n_features=n_features,
            hidden_size=params.get("hidden_size", 128),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.2),
        ).to(device)
    else:
        raise NotImplementedError(
            "TFT model requires pytorch-forecasting TimeSeriesDataSet. "
            "Set model: 'lstm' for quick training, or see the TFT section of the README."
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # ── Training ──────────────────────────────────────────────────────────────
    max_epochs = params.get("max_epochs", 50)
    best_mae   = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).cpu().numpy()
                preds_all.extend(pred)
                true_all.extend(yb.numpy())

        val_mae = mae(true_all, preds_all)
        scheduler.step(val_mae)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  train_loss=%.4f  val_MAE=%.4f pp",
                epoch, max_epochs, train_loss / len(train_loader), val_mae,
            )

        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "mae": val_mae},
                path=cfg.get("checkpoint", "checkpoints/gdp_nowcast_best.ckpt"),
            )

    log.info("Best val MAE: %.4f pp", best_mae)
    log.info("Pearson r:    %.4f", pearson_r(true_all, preds_all))
