"""
Experiment: Central Bank NLP
==============================
Fine-tunes FinBERT on FOMC / ECB / BOE meeting minutes to classify
sentence segments as hawkish | neutral | dovish.

Post-training, builds a monthly sentiment index and plots it against
yield-curve changes as a downstream validation.

Config keys (under exp_params):
    model_name:    "ProsusAI/finbert"
    freeze_base:   false
    dropout:       0.1
    lr:            2e-5
    batch_size:    32
    max_epochs:    5
    max_length:    256
    sources:       ["fomc", "ecb", "boe"]
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.trainer import register
from src.utils.io import get_device, set_seed, save_checkpoint
from src.utils.metrics import macro_f1, confusion_matrix_str

log = logging.getLogger(__name__)

LABEL_NAMES = ["hawkish", "neutral", "dovish"]


@register("central_bank_nlp")
def run(cfg: dict) -> None:
    params = cfg.get("exp_params", {})
    smoke  = params.get("smoke", False)
    device = get_device(params.get("device"))
    set_seed(params.get("seed", 42))

    model_name  = params.get("model_name", "ProsusAI/finbert")
    batch_size  = params.get("batch_size", 32)
    max_length  = params.get("max_length", 256)
    max_epochs  = params.get("max_epochs", 5)
    lr          = params.get("lr", 2e-5)

    log.info("Model: %s | Device: %s | Smoke: %s", model_name, device, smoke)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if smoke:
        from src.datasets.fomc import FOMCDataset
        dataset = FOMCDataset.synthetic(tokenizer, n=128, max_length=max_length)
        log.info("Using synthetic dataset (%d samples)", len(dataset))
    else:
        from src.datasets.fomc import FOMCDataset
        dataset = FOMCDataset(
            data_dir=cfg.get("data", {}).get("fomc_dir", "data/fomc"),
            tokenizer=tokenizer,
            max_length=max_length,
            sources=params.get("sources", ["fomc", "ecb", "boe"]),
        )

    n_val   = max(16, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.models.nlp import CentralBankClassifier
    model = CentralBankClassifier(
        model_name=model_name,
        num_labels=3,
        dropout=params.get("dropout", 0.1),
        freeze_base=params.get("freeze_base", False),
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=params.get("weight_decay", 1e-4),
    )
    total_steps = max_epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_f1 = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        f1 = macro_f1(all_labels, all_preds)
        log.info("Epoch %2d/%d  loss=%.4f  val macro-F1=%.4f", epoch, max_epochs, avg_loss, f1)

        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "f1": f1},
                path=cfg.get("checkpoint", "checkpoints/central_bank_nlp_best.ckpt"),
            )

    log.info("Best val macro-F1: %.4f", best_f1)

    # Confusion matrix
    log.info(
        "Confusion matrix:\n%s",
        confusion_matrix_str(all_labels, all_preds, LABEL_NAMES),
    )
