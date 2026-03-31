"""
Experiment: Informal Economy Detection
========================================
Trains a ResNet-18 regression model on VIIRS annual nightlight composite tiles
to estimate subnational log GDP per capita as a proxy for formal / informal
economic activity levels.

Config keys (under exp_params):
    in_channels:   1
    pretrained:    true
    dropout:       0.3
    lr:            1e-4
    batch_size:    32
    max_epochs:    30
    tile_size:     256
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.trainer import register
from src.utils.io import get_device, set_seed, save_checkpoint
from src.utils.metrics import mae, rmse, pearson_r

log = logging.getLogger(__name__)


@register("informal_economy")
def run(cfg: dict) -> None:
    params     = cfg.get("exp_params", {})
    smoke      = params.get("smoke", False)
    device     = get_device(params.get("device"))
    set_seed(params.get("seed", 42))
    batch_size = params.get("batch_size", 32)
    max_epochs = params.get("max_epochs", 30)

    log.info("Device: %s | Smoke: %s", device, smoke)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if smoke:
        from src.datasets.nightlights import NightlightTileDataset
        train_ds = NightlightTileDataset.synthetic(n=64, tile_size=64)
        val_ds   = NightlightTileDataset.synthetic(n=16, tile_size=64)
        in_channels = 1
        log.info("Using synthetic tiles for smoke test")
    else:
        from src.datasets.nightlights import NightlightTileDataset
        data_dir    = cfg.get("data", {}).get("nightlights_dir", "data/nightlights")
        in_channels = params.get("in_channels", 1)
        train_ds = NightlightTileDataset(data_dir, split="train")
        val_ds   = NightlightTileDataset(data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.models.cnn import NightlightRegressor
    model = NightlightRegressor(
        in_channels=params.get("in_channels", in_channels),
        pretrained=params.get("pretrained", not smoke),
        dropout=params.get("dropout", 0.3),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.get("lr", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # ── Training ──────────────────────────────────────────────────────────────
    best_mae = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x   = batch["x"].to(device)
            y   = batch["target"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["x"].to(device)).cpu().numpy()
                preds_all.extend(pred)
                true_all.extend(batch["target"].numpy())

        val_mae = mae(true_all, preds_all)

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  train_loss=%.4f  val_MAE=%.4f",
                epoch, max_epochs, train_loss / len(train_loader), val_mae,
            )

        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "mae": val_mae},
                path=cfg.get("checkpoint", "checkpoints/informal_economy_best.ckpt"),
            )

    log.info("Best val MAE: %.4f | Pearson r: %.4f",
             best_mae, pearson_r(true_all, preds_all))
