"""
Experiment: Economic Complexity GNN
====================================
Trains a GraphSAGE model on the OEC product-space bipartite graph to:
  1. Regress country ECI scores  (supervised, MSE loss)
  2. Predict future export links  (link prediction, BCE loss)

Config keys (under exp_params):
    hidden_channels:  128
    out_channels:     64
    num_layers:       3
    dropout:          0.2
    lr:               1e-3
    max_epochs:       100
    weight_decay:     1e-4
    oec_year:         2019
    min_rca:          1.0
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from src.trainer import register
from src.utils.io import get_device, set_seed
from src.utils.metrics import ndcg_at_k, pearson_r

log = logging.getLogger(__name__)


@register("economic_complexity")
def run(cfg: dict) -> None:
    params     = cfg.get("exp_params", {})
    smoke      = params.get("smoke", False)
    device     = get_device(params.get("device"))
    seed       = params.get("seed", 42)
    set_seed(seed)

    log.info("Device: %s | Smoke: %s", device, smoke)

    # ── Data ─────────────────────────────────────────────────────────────────
    if smoke:
        from src.datasets.oec import OECDataset
        data = OECDataset.synthetic_graph(n_countries=30, n_products=60)
        data.y_eci = torch.randn(data.country_mask.sum())
        log.info("Using synthetic graph for smoke test")
    else:
        from src.datasets.oec import OECDataset
        ds   = OECDataset(
            data_dir=cfg.get("data", {}).get("oec_dir", "data/oec"),
            year=params.get("oec_year", 2019),
            min_rca=params.get("min_rca", 1.0),
        )
        data = ds.build_graph()

    data = data.to(device)
    n_nodes = data.x.shape[0]

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.models.gnn import ProductSpaceGNN, ECIRegressor, LinkPredictor

    gnn = ProductSpaceGNN(
        in_channels=n_nodes,
        hidden_channels=params.get("hidden_channels", 128),
        out_channels=params.get("out_channels", 64),
        num_layers=params.get("num_layers", 3),
        dropout=params.get("dropout", 0.2),
    ).to(device)

    regressor      = ECIRegressor(gnn, embedding_dim=params.get("out_channels", 64)).to(device)
    link_predictor = LinkPredictor(embedding_dim=params.get("out_channels", 64)).to(device)

    optimizer = torch.optim.AdamW(
        list(regressor.parameters()) + list(link_predictor.parameters()),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.get("max_epochs", 100)
    )

    # ── Negative sampling helper ──────────────────────────────────────────────
    def sample_negatives(edge_index, n_nodes, n_neg):
        neg_src = torch.randint(0, edge_index.max().item(), (n_neg,), device=device)
        neg_dst = torch.randint(0, n_nodes, (n_neg,), device=device)
        return neg_src, neg_dst

    # ── Training loop ─────────────────────────────────────────────────────────
    max_epochs = params.get("max_epochs", 100)
    log.info("Training for %d epochs …", max_epochs)

    for epoch in range(1, max_epochs + 1):
        regressor.train()
        link_predictor.train()
        optimizer.zero_grad()

        # ECI regression loss
        emb     = gnn(data.x, data.edge_index)
        n_c     = data.country_mask.sum().item()
        country_emb = emb[data.country_mask]

        if hasattr(data, "y_eci") and data.y_eci is not None:
            y_eci = data.y_eci.to(device)
            eci_pred = regressor.head(country_emb).squeeze(-1)
            loss_eci = F.mse_loss(eci_pred, y_eci)
        else:
            loss_eci = torch.tensor(0.0, device=device)

        # Link prediction loss
        pos_src = data.edge_index[0]
        pos_dst = data.edge_index[1]
        n_pos   = pos_src.shape[0]
        neg_src, neg_dst = sample_negatives(data.edge_index, data.x.shape[0], n_pos)

        pos_scores = link_predictor(emb, pos_src, pos_dst)
        neg_scores = link_predictor(emb, neg_src, neg_dst)

        labels = torch.cat([
            torch.ones(n_pos, device=device),
            torch.zeros(n_pos, device=device),
        ])
        scores = torch.cat([pos_scores, neg_scores])
        loss_link = F.binary_cross_entropy_with_logits(scores, labels)

        loss = loss_eci + loss_link
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(regressor.parameters()) + list(link_predictor.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  loss=%.4f  (eci=%.4f  link=%.4f)",
                epoch, max_epochs, loss.item(), loss_eci.item(), loss_link.item(),
            )

    log.info("Training complete.")

    # ── Quick eval ────────────────────────────────────────────────────────────
    regressor.eval()
    with torch.no_grad():
        emb = gnn(data.x, data.edge_index)
        if hasattr(data, "y_eci") and data.y_eci is not None:
            eci_pred = regressor.head(emb[data.country_mask]).squeeze(-1)
            r = pearson_r(data.y_eci.cpu().numpy(), eci_pred.cpu().numpy())
            log.info("Pearson r (ECI regression): %.4f", r)
