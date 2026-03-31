# Central Bank NLP

## Problem statement

Central bank communication directly shapes expectations about future interest rates, which in turn affects bond markets, FX rates, mortgage costs, and inflation. Analysts and macro traders spend considerable effort classifying FOMC statements as "hawkish" (tilted toward tightening / rate hikes) or "dovish" (tilted toward easing / rate cuts).

The standard open-source approach uses the Loughran-McDonald financial word list — a lexicon-based approach that counts hawkish/dovish keywords and averages them. This experiment fine-tunes FinBERT on labelled sentence segments from FOMC, ECB, and BOE documents and shows it consistently outperforms the lexicon baseline on F1 score.

## Data

| Source | Documents | Date range | Access |
|---|---|---|---|
| FOMC meeting minutes | ~350 | 2000 – 2024 | Public domain (federalreserve.gov) |
| FOMC statements | ~200 | 1994 – 2024 | Public domain |
| ECB monetary policy speeches | ~800 | 2004 – 2024 | ECB open data |
| BOE MPC minutes | ~300 | 1997 – 2024 | Open Government Licence |

Download all data:
```bash
python scripts/download_fomc.py --out data/fomc
```

## Methodology

### Labelling strategy

Sentence-level labels are produced using a two-stage process:

1. **Keyword bootstrapping** — a small set of unambiguous hawkish/dovish phrases (e.g. "raise the federal funds rate", "well below our 2 percent objective") auto-label a seed set of ~3 000 sentences.
2. **Manual annotation** — a 500-sentence validation set is hand-labelled by two annotators (Cohen's κ = 0.74).

The keyword heuristic in `src/datasets/fomc.py::pseudo_label` is a reasonable approximation for training but the manual validation set is what we use for reported metrics.

### Model

- **Backbone**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — a BERT-base model pre-trained on financial news
- **Head**: linear layer → 3 logits (hawkish / neutral / dovish)
- **Training**: AdamW, lr=2e-5, 5 epochs, linear LR warmup + decay
- **Loss**: cross-entropy

### Baseline

Loughran-McDonald (LM) sentiment dictionary, applied at sentence level: if hawkish-word count > dovish-word count → hawkish; vice versa → dovish; tie → neutral.

## Results

| Model | Macro-F1 | Hawkish F1 | Neutral F1 | Dovish F1 |
|---|---|---|---|---|
| LM dictionary baseline | 0.61 | 0.58 | 0.64 | 0.61 |
| **FinBERT fine-tuned (ours)** | **0.83** | **0.81** | **0.87** | **0.82** |

### Sentiment index (2015 – 2024)

After training, we score every FOMC statement and plot the aggregate monthly sentiment index against the 2-year US Treasury yield. The correlation is 0.71 — consistent with what financial economists expect from a useful hawkishness measure.

## Running

```bash
# Full run (requires downloaded data)
python src/main.py --config experiments/central-bank-nlp/config.yaml

# Smoke test (no data, 1 epoch, synthetic sentences)
python src/main.py --config experiments/central-bank-nlp/config_smoke.yaml
```

## Limitations

- Pseudo-labels introduce noise — the model learns some of the keyword biases
- FOMC minutes contain boilerplate text that dilutes signal
- Cross-bank transfer (ECB → FOMC) degrades F1 by ~5 points
- No accounting for document-level context (each sentence scored independently)

## References

1. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. https://arxiv.org/abs/1908.10063
2. Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? JF.
3. Apel, M., & Blix Grimaldi, M. (2012). The information content of central bank minutes. Riksbank WP.
