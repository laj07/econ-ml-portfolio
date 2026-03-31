"""
Experiment: Labour Market Mismatch
=====================================
Embeds ESCO skill labels and job-posting descriptions into a shared semantic
space using Sentence-BERT, then computes supply/demand mismatch metrics and
produces a UMAP visualisation of the skill landscape.

Config keys (under exp_params):
    model_name:    "all-MiniLM-L6-v2"
    batch_size:    256
    umap_n_neighbors: 15
    umap_min_dist:    0.1
    mismatch_threshold: 0.70
"""
from __future__ import annotations

import logging

import numpy as np

from src.trainer import register
from src.utils.io import get_device, set_seed

log = logging.getLogger(__name__)


@register("labor_market")
def run(cfg: dict) -> None:
    params     = cfg.get("exp_params", {})
    smoke      = params.get("smoke", False)
    device_str = get_device(params.get("device")).type
    set_seed(params.get("seed", 42))

    model_name = params.get("model_name", "all-MiniLM-L6-v2")
    log.info("Embedder: %s | Device: %s | Smoke: %s", model_name, device_str, smoke)

    # ── Data ─────────────────────────────────────────────────────────────────
    if smoke:
        from src.datasets.esco import JobPostingsDataset
        postings = JobPostingsDataset.synthetic(n_postings=50)
        supply_labels = [
            "Python", "SQL", "Data Analysis", "Machine Learning",
            "Project Management", "Communication", "Agile",
            "Statistical Modelling", "R", "Java",
        ]
        supply_descs = supply_labels  # same for smoke
        log.info("Using synthetic data for smoke test")
    else:
        from src.datasets.esco import ESCOSkills, JobPostingsDataset
        esco    = ESCOSkills(cfg.get("data", {}).get("esco_dir", "data/esco"))
        postings = JobPostingsDataset(
            postings_csv=cfg.get("data", {}).get("postings_csv", "data/postings.csv"),
            esco=esco,
        )
        supply_labels = esco.skill_labels()
        supply_descs  = esco.skill_descriptions()

    # ── Embed ─────────────────────────────────────────────────────────────────
    from src.models.embeddings import SkillEmbedder, MismatchAnalyser

    embedder = SkillEmbedder(model_name=model_name, device=device_str)

    demand_texts  = postings.posting_texts()
    demand_labels = [f"job_{i}" for i in range(len(demand_texts))]

    log.info("Embedding %d job postings …", len(demand_texts))
    demand_embs = embedder.embed(demand_texts, show_progress=not smoke)

    log.info("Embedding %d ESCO skills …", len(supply_labels))
    supply_embs = embedder.embed(supply_descs, show_progress=not smoke)

    # ── Mismatch analysis ─────────────────────────────────────────────────────
    threshold = params.get("mismatch_threshold", 0.70)
    analyser  = MismatchAnalyser(demand_embs, supply_embs, demand_labels, supply_labels)

    score = analyser.mismatch_score(threshold=threshold)
    log.info("Aggregate mismatch score (frac unmatched @ %.2f): %.4f", threshold, score)

    top_orphan_demand = analyser.orphan_demand(threshold=threshold, top_n=10)
    top_orphan_supply = analyser.orphan_supply(threshold=threshold, top_n=10)

    log.info("Most under-supplied demanded skills:\n  %s", "\n  ".join(top_orphan_demand))
    log.info("Most under-demanded supplied skills:\n  %s", "\n  ".join(top_orphan_supply))

    # ── UMAP projection ───────────────────────────────────────────────────────
    try:
        import umap

        log.info("Computing UMAP projection …")
        all_embs   = np.vstack([demand_embs, supply_embs])
        all_labels = (
            ["demand"] * len(demand_embs) + ["supply"] * len(supply_embs)
        )

        reducer   = umap.UMAP(
            n_neighbors=params.get("umap_n_neighbors", 15),
            min_dist=params.get("umap_min_dist", 0.1),
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        embs_2d   = reducer.fit_transform(all_embs)

        fig_path = cfg.get("figures", {}).get(
            "umap", "output/labor_market_umap.png"
        )
        from src.utils.viz import plot_umap_embeddings
        plot_umap_embeddings(
            embs_2d, all_labels,
            title="Supply vs demand skill landscape (UMAP)",
            save_path=fig_path,
        )
        log.info("UMAP figure saved → %s", fig_path)

    except ImportError:
        log.warning("umap-learn not installed — skipping UMAP projection. pip install umap-learn")

    log.info("Labour market mismatch experiment complete.")
