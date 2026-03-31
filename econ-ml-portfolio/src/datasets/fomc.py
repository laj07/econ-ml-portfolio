"""
FOMC / ECB / BOE minutes dataset for central-bank NLP.

Downloads and parses meeting minutes and speeches into labelled sentence
segments. Labels come from a small hand-annotated seed set + keyword rules
(same approach as Apel & Blix Grimaldi 2012).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Keyword heuristics (used to create pseudo-labels for sentences not in seed set)
_HAWKISH_TOKENS = frozenset([
    "inflationary", "overheating", "tightening", "hike", "hikes",
    "above target", "raise rates", "upside risk", "price pressures",
])
_DOVISH_TOKENS = frozenset([
    "accommodation", "accommodative", "easing", "cut rates", "below target",
    "downside risk", "slack", "deflationary", "support growth",
])


def pseudo_label(text: str) -> str:
    """Assign a coarse label using keyword matching. For bootstrapping only."""
    lower = text.lower()
    h = sum(1 for t in _HAWKISH_TOKENS if t in lower)
    d = sum(1 for t in _DOVISH_TOKENS  if t in lower)
    if h > d:
        return "hawkish"
    if d > h:
        return "dovish"
    return "neutral"


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter adequate for formal policy prose."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.split()) >= 5]


class FOMCDataset(Dataset):
    """
    PyTorch Dataset over FOMC (and optionally ECB/BOE) document segments.

    Args:
        data_dir:    directory with .txt files (one document per file)
        tokenizer:   HuggingFace tokenizer
        max_length:  token limit per segment
        label_col:   column name for labels ('label' or manual annotation)
        sources:     which central banks to include
    """

    LABEL2ID = {"hawkish": 0, "neutral": 1, "dovish": 2}

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer,
        max_length: int = 256,
        sources: list[str] | None = None,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.sources    = sources or ["fomc", "ecb", "boe"]

        self.records = self._load(Path(data_dir))
        log.info(
            "FOMCDataset: %d segments from %s",
            len(self.records), self.sources
        )

    def _load(self, data_dir: Path) -> list[dict]:
        records = []
        for src in self.sources:
            src_dir = data_dir / src
            if not src_dir.exists():
                log.warning("Source directory missing: %s — skipping", src_dir)
                continue
            for fpath in sorted(src_dir.glob("*.txt")):
                text = fpath.read_text(errors="replace")
                for sent in split_sentences(text):
                    records.append({
                        "text":   sent,
                        "label":  pseudo_label(sent),
                        "source": src,
                        "doc":    fpath.stem,
                    })
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = self.LABEL2ID[rec["label"]]
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }

    # ── Smoke-test synthetic ──────────────────────────────────────────────────

    @staticmethod
    def synthetic(tokenizer, n: int = 64, max_length: int = 64) -> "FOMCDataset":
        """Return a tiny synthetic dataset for smoke testing."""
        import types

        texts = (
            ["The committee decided to raise interest rates by 25 basis points."] * (n // 3)
            + ["The outlook remains accommodative with rates held steady."] * (n // 3)
            + ["Inflation remains well below the 2 percent target."] * (n - 2 * (n // 3))
        )
        labels = (
            ["hawkish"] * (n // 3)
            + ["neutral"] * (n // 3)
            + ["dovish"] * (n - 2 * (n // 3))
        )
        records = [{"text": t, "label": l, "source": "synthetic", "doc": "smoke"}
                   for t, l in zip(texts, labels)]

        ds = FOMCDataset.__new__(FOMCDataset)
        ds.tokenizer  = tokenizer
        ds.max_length = max_length
        ds.sources    = ["synthetic"]
        ds.records    = records
        return ds
