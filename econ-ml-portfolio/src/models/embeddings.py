"""
Sentence-BERT wrapper and skills embedding model for labour-market analysis.

Embeds job descriptions and ESCO skill labels into a shared semantic space,
then quantifies supply/demand mismatch via cosine similarity and UMAP clustering.

References
----------
Reimers & Gurevych (2019) Sentence-BERT. EMNLP.
https://arxiv.org/abs/1908.10084
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SkillEmbedder:
    """
    Wraps sentence-transformers to embed job postings and ESCO skill labels.

    Args:
        model_name: any SentenceTransformer model ID.
                    'all-MiniLM-L6-v2' is fast; 'all-mpnet-base-v2' is stronger.
        device:     'cpu', 'cuda', or 'mps'
        batch_size: embedding batch size
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 256,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        self.model      = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.device     = device

    def embed(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, D)."""
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embs

    def pairwise_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Cosine similarity matrix between two embedding sets. Shape: (|a|, |b|)."""
        return (a @ b.T).clip(-1, 1)

    def top_k_skills(
        self,
        job_emb: np.ndarray,
        skill_embs: np.ndarray,
        skill_labels: list[str],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the top-k closest ESCO skills for a single job embedding."""
        sims = (job_emb @ skill_embs.T).flatten()
        top  = np.argsort(sims)[::-1][:k]
        return [(skill_labels[i], float(sims[i])) for i in top]


class MismatchAnalyser:
    """
    Computes aggregate supply/demand mismatch metrics.

    demand_embs  — embeddings of required skills extracted from job postings
    supply_embs  — embeddings of self-reported skills from CVs / ESCO taxonomy
    """

    def __init__(
        self,
        demand_embs: np.ndarray,
        supply_embs: np.ndarray,
        demand_labels: list[str],
        supply_labels: list[str],
    ) -> None:
        self.demand_embs   = demand_embs
        self.supply_embs   = supply_embs
        self.demand_labels = demand_labels
        self.supply_labels = supply_labels

    def mismatch_score(self, threshold: float = 0.7) -> float:
        """
        Fraction of demanded skills with no close supply-side match.
        Higher = more mismatch.
        """
        sim = self.demand_embs @ self.supply_embs.T   # (D, S)
        max_sim = sim.max(axis=1)                     # best supply match for each demand
        return float((max_sim < threshold).mean())

    def orphan_demand(self, threshold: float = 0.7, top_n: int = 20) -> list[str]:
        """Return the top-N demanded skills with the weakest supply coverage."""
        sim = self.demand_embs @ self.supply_embs.T
        max_sim = sim.max(axis=1)
        idx = np.argsort(max_sim)[:top_n]
        return [self.demand_labels[i] for i in idx]

    def orphan_supply(self, threshold: float = 0.7, top_n: int = 20) -> list[str]:
        """Return the top-N supplied skills with the weakest demand coverage."""
        sim = self.supply_embs @ self.demand_embs.T
        max_sim = sim.max(axis=1)
        idx = np.argsort(max_sim)[:top_n]
        return [self.supply_labels[i] for i in idx]
