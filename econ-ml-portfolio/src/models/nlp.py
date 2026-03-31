"""
FinBERT-based classifier for central bank communication.

Classifies text segments from FOMC, ECB, and BOE documents as:
  hawkish (0) | neutral (1) | dovish (2)

References
----------
Araci (2019) FinBERT: Financial Sentiment Analysis with Pre-trained LMs
https://arxiv.org/abs/1908.10063
"""
from __future__ import annotations

import torch
import torch.nn as nn


LABEL2ID = {"hawkish": 0, "neutral": 1, "dovish": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}


class CentralBankClassifier(nn.Module):
    """
    FinBERT (ProsusAI/finbert) with a 3-class head.

    Args:
        model_name:  any HuggingFace encoder; defaults to ProsusAI/finbert
        num_labels:  number of sentiment classes
        dropout:     dropout on the [CLS] embedding before the head
        freeze_base: freeze all BERT weights (train head only)
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        num_labels: int = 3,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = self.drop(out.last_hidden_state[:, 0, :])
        return self.head(cls)                # (B, num_labels)


class SentimentIndex:
    """
    Aggregates sentence-level predictions into a document-level
    net-dovish score in [-1, +1].  Positive = dovish, negative = hawkish.
    """

    def __init__(
        self,
        model: CentralBankClassifier,
        tokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def score_sentences(self, sentences: list[str]) -> list[dict]:
        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        probs = torch.softmax(self.model(**enc), dim=-1).cpu().numpy()

        results = []
        for i, sent in enumerate(sentences):
            label = ID2LABEL[int(probs[i].argmax())]
            results.append({
                "text":         sent,
                "label":        label,
                "hawkish_prob": float(probs[i][0]),
                "neutral_prob": float(probs[i][1]),
                "dovish_prob":  float(probs[i][2]),
                "score":        float(probs[i][2] - probs[i][0]),   # net dovish
            })
        return results

    def document_score(self, sentences: list[str]) -> float:
        scored = [s for s in self.score_sentences(sentences) if s["label"] != "neutral"]
        if not scored:
            return 0.0
        return sum(s["score"] for s in scored) / len(scored)
