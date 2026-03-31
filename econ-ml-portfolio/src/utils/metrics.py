"""
Evaluation metrics used across experiments.

All functions accept plain numpy arrays or torch tensors (converted internally).
"""

from __future__ import annotations

import numpy as np


def to_numpy(x) -> np.ndarray:
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


# ── Information retrieval ─────────────────────────────────────────────────────

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain @k.

    Args:
        y_true:  binary relevance labels (1 = relevant, 0 = not)
        y_score: predicted scores (higher = more likely to be relevant)
        k:       rank cutoff

    Returns:
        NDCG@k in [0, 1]
    """
    y_true = to_numpy(y_true).astype(float)
    y_score = to_numpy(y_score).astype(float)

    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order[:k]]

    gains = y_true_sorted / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()

    # Ideal DCG
    ideal_sorted = np.sort(y_true)[::-1][:k]
    idcg = (ideal_sorted / np.log2(np.arange(2, k + 2))).sum()

    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean Reciprocal Rank (MRR)."""
    y_true = to_numpy(y_true).astype(float)
    y_score = to_numpy(y_score).astype(float)
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0


# ── Regression ────────────────────────────────────────────────────────────────

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(to_numpy(y_true) - to_numpy(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((to_numpy(y_true) - to_numpy(y_pred)) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def pearson_r(y_true, y_pred) -> float:
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ── Classification ────────────────────────────────────────────────────────────

def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(to_numpy(y_true), to_numpy(y_pred), average="macro"))


def confusion_matrix_str(y_true, y_pred, labels: list[str]) -> str:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(to_numpy(y_true), to_numpy(y_pred))
    header = "        " + "  ".join(f"{l:>8}" for l in labels)
    rows = [header]
    for i, row in enumerate(cm):
        rows.append(f"{labels[i]:>8}  " + "  ".join(f"{v:>8}" for v in row))
    return "\n".join(rows)
