"""Plotting helpers shared across experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "primary":   "#5346B7",
    "secondary": "#1D9E75",
    "accent":    "#EF9F27",
    "danger":    "#E24B4A",
    "neutral":   "#888780",
}


def save_or_show(path: str | Path | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Training curve",
    ylabel: str = "Loss",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train", color=PALETTE["primary"])
    if val_losses:
        ax.plot(val_losses, label="val", color=PALETTE["secondary"], linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_or_show(save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Confusion matrix",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_or_show(save_path)


def plot_sentiment_index(
    dates: list,
    scores: list[float],
    title: str = "Monetary policy sentiment index",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [PALETTE["secondary"] if s >= 0 else PALETTE["danger"] for s in scores]
    ax.bar(dates, scores, color=colors, width=20)
    ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8, linestyle="--")
    ax.set_ylabel("Dovish (+) / Hawkish (−)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_or_show(save_path)


def plot_umap_embeddings(
    embeddings_2d: np.ndarray,
    labels: list[str],
    title: str = "UMAP projection",
    save_path: str | Path | None = None,
) -> None:
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    label2idx = {l: i for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=[cmap(label2idx[label])], label=label, s=10, alpha=0.7,
        )
    ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_or_show(save_path)
