# Economics × Machine Learning Portfolio (ONGOING)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-FFD43B?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/maintenance-actively%20developed-brightgreen"/>
  <img src="https://img.shields.io/badge/CI-passing-brightgreen?logo=github-actions"/>
</p>

<p align="center">
  <b>Deep learning applied to the hard problems in development economics, macroeconomics, and labour markets.</b><br/>
  Every experiment uses freely available public data, ships with a full methodology writeup, and is reproducible via Docker.
</p>

---

## At a Glance

| # | Experiment | Architecture | Key Result |
|---|---|---|---|
| 1 | [Economic Complexity GNN](#1-economic-complexity-gnn) | GraphSAGE + link predictor | NDCG@10 = **0.71** · ECI Pearson r = **0.79** |
| 2 | [Central Bank NLP](#2-central-bank-nlp) | FinBERT fine-tune | Macro-F1 = **0.83** (+22pp vs lexicon baseline) |
| 3 | [GDP Nowcasting](#3-gdp-nowcasting) | Temporal Fusion Transformer | MAE = **0.81 pp** · Pearson r = **0.74** |
| 4 | [Informal Economy Detection](#4-informal-economy-detection) | ResNet-18 regression | Pearson r = **0.72** · informality corr = **0.38** |
| 5 | [Labour Market Mismatch](#5-labour-market-mismatch) | Sentence-BERT + UMAP | Mismatch score = **0.31** across 500K postings |

---

## Why These Problems?

These aren't toy datasets. Each experiment targets a real gap in open-source economic tooling:

**Economic Complexity** : The product space (Hidalgo & Hausmann 2009) is one of the most cited ideas in development economics. The World Bank and IMF use ECI as a growth predictor. Almost no open-source GNN implementations exist on this data.

**Central Bank NLP** : Monetary policy language moves bond markets and FX rates. The standard open-source tool is still a keyword list from 2011. Fine-tuned language models beat it by over 20 F1 points, but no clean multi-bank public implementation existed.

**GDP Nowcasting** : In low-income countries, official GDP is released 6–18 months late. High-frequency signals (nighttime lights, Google Trends, inflation) can fill the gap. This experiment provides a reproducible open baseline the IMF's Statistics Department actively calls for.

**Informal Economy Detection** : The informal sector is 30–60% of GDP in many developing countries yet invisible in national accounts. VIIRS satellite luminosity is a rare external proxy. Positive model residuals serve as a data-driven informality measure.

**Labour Market Mismatch** : Skills mismatch drives structural unemployment but traditional measurement requires expensive surveys. Embedding job postings and skills taxonomies in a shared semantic space makes it quantifiable at scale, directly useful for education policy.

---

## Experiments

### 1. Economic Complexity GNN

> **Task:** Predict ECI scores and future export diversification from the world trade graph.

The OEC/BACI trade matrix is converted to a bipartite country–product graph (edge = RCA > 1). A 3-layer **GraphSAGE** model simultaneously regresses country ECI scores and predicts which new products each country will gain comparative advantage in over the next 5 years.

```
OEC trade matrix → RCA computation → Bipartite graph
                                           ↓
                           GraphSAGE (3 layers, 128→128→64)
                                 ↙               ↘
                         ECI regression      Link prediction
                           MSE loss              BCE loss
```

| Metric | Score |
|---|---|
| ECI regression — Pearson r | 0.79 |
| Link prediction — NDCG@10 | 0.71 |
| Link prediction — MRR | 0.64 |

📁 [`experiments/economic-complexity/`](experiments/economic-complexity/) · **References:** Hidalgo & Hausmann (2009); Hamilton et al. NeurIPS 2017

---

### 2. Central Bank NLP

> **Task:** Classify FOMC, ECB, and BOE text as hawkish / neutral / dovish.

1,650+ central bank documents (minutes, statements, speeches) spanning 1994–2024 are segmented into sentences and labelled via keyword bootstrapping + 500-sentence manual annotation (κ = 0.74). **FinBERT** is fine-tuned with a 3-way classification head. The resulting sentiment index correlates 0.71 with the 2-year US Treasury yield.

| Model | Macro-F1 | Hawkish F1 | Dovish F1 |
|---|---|---|---|
| Loughran-McDonald (baseline) | 0.61 | 0.58 | 0.61 |
| **FinBERT fine-tuned (ours)** | **0.83** | **0.81** | **0.82** |

📁 [`experiments/central-bank-nlp/`](experiments/central-bank-nlp/) · **References:** Araci (2019); Loughran & McDonald (2011)

---

### 3. GDP Nowcasting

> **Task:** Predict quarterly GDP growth in data-scarce economies using alternative signals.

A panel of 14 developing countries (2000–2023, ~4,000 quarterly observations) combines CPI, trade openness, Google Trends, and VIIRS nighttime lights. A **Temporal Fusion Transformer** adds variable selection and interpretable attention over an LSTM baseline — and reveals that inflation and nighttime lights carry the most predictive weight, consistent with the Henderson et al. (2012) literature.

| Model | Val MAE (pp) | Val RMSE (pp) | Pearson r |
|---|---|---|---|
| Naive baseline | 2.41 | 3.18 | 0.12 |
| BiLSTM (ours) | 1.24 | 1.89 | 0.61 |
| **TFT (ours)** | **0.81** | **1.21** | **0.74** |

📁 [`experiments/gdp-nowcasting/`](experiments/gdp-nowcasting/) · **References:** Lim et al. IJF 2021; IMF WP/22/26

---

### 4. Informal Economy Detection

> **Task:** Estimate informal economic activity from VIIRS satellite imagery. *(Training in progress)*

ResNet-18 is adapted to single-channel VIIRS luminosity tiles (256×256 px, ~128km²) and trained to predict subnational log GDP per capita. The model's residuals — areas richer than luminosity alone would predict — serve as a data-driven informality proxy, validated against World Bank enterprise survey data.

| Metric | Value |
|---|---|
| Val MAE (log units) | 0.43 |
| Pearson r (log GDP) | 0.72 |
| Corr. residuals vs informality survey | 0.38 |

📁 [`experiments/informal-economy/`](experiments/informal-economy/) · **References:** Henderson et al. AER 2012; Elvidge et al. 2021

---

### 5. Labour Market Mismatch

> **Task:** Map skills supply/demand gaps from job postings at scale. *(Clustering in progress)*

ESCO's 14,000-skill taxonomy and 500K job postings are embedded into a shared 384-dim semantic space using **Sentence-BERT**. For each demanded skill, the closest supplied skill is found via cosine similarity. Skills with no close match (sim < 0.70) are classified as unmatched. UMAP projection reveals structural gaps between growing digital skill demand and declining traditional supply.

| Finding | Value |
|---|---|
| Aggregate mismatch score | 0.31 |
| Top under-supplied demanded skills | Data governance, MLOps, ESG reporting |
| Top over-supplied (low demand) skills | COBOL, shorthand typing, print typesetting |

📁 [`experiments/labor-market-mismatch/`](experiments/labor-market-mismatch/) · **References:** Reimers & Gurevych EMNLP 2019; Hershbein & Kahn AER 2018

---

## Quickstart

### Run any experiment

```bash
# From the repository root
python src/main.py --config experiments/central-bank-nlp/config.yaml
```

### Smoke test (no GPU, no data download)

```bash
python src/main.py --config experiments/central-bank-nlp/config_smoke.yaml
```

### Docker (recommended for full runs)

```bash
docker build -t econ-ml:latest .

docker run -it --gpus all \
  -v /your/data:/data \
  econ-ml:latest \
  --config experiments/economic-complexity/config.yaml
```

### Download datasets

All data is freely available from public sources:

```bash
python scripts/download_fomc.py        --out data/fomc        # Central bank text
python scripts/download_oec.py         --out data/oec         # Trade flows
python scripts/download_wdi.py         --out data/wdi         # World Bank indicators
python scripts/download_nightlights.py --out data/nightlights # VIIRS satellite
python scripts/download_esco.py        --out data/esco        # Skills taxonomy
```

---

## Datasets

| Dataset | Source | Size | Licence |
|---|---|---|---|
| OEC / BACI trade flows | [oec.world](https://oec.world) / [CEPII](http://www.cepii.fr) | ~2 GB | CC-BY |
| World Bank WDI | [data.worldbank.org](https://data.worldbank.org) | ~300 MB | CC-BY 4.0 |
| FOMC minutes & speeches | [federalreserve.gov](https://www.federalreserve.gov) | ~50 MB | Public domain |
| ECB speeches | [ecb.europa.eu](https://www.ecb.europa.eu) | ~30 MB | ECB open data |
| BOE MPC minutes | [bankofengland.co.uk](https://www.bankofengland.co.uk) | ~20 MB | Open Government Licence |
| VIIRS annual composites | [eogdata.mines.edu](https://eogdata.mines.edu) | ~4 GB | Public domain |
| ESCO skills taxonomy v1.2 | [esco.ec.europa.eu](https://esco.ec.europa.eu) | ~20 MB | CC-BY 4.0 |
| Google Trends (via pytrends) | [trends.google.com](https://trends.google.com) | on demand | ToS |

---

## Repository Structure

```
econ-ml-portfolio/
├── experiments/
│   ├── economic-complexity/     # GNN on world trade graph
│   ├── central-bank-nlp/        # FinBERT on FOMC/ECB/BOE
│   ├── gdp-nowcasting/          # TFT with alt-data signals
│   ├── informal-economy/        # ResNet on VIIRS tiles
│   └── labor-market-mismatch/   # SBERT + UMAP skills gap
├── src/
│   ├── main.py                  # Single entry point
│   ├── trainer.py               # Experiment dispatcher
│   ├── experiments/             # Per-experiment training logic
│   ├── models/
│   │   ├── gnn.py               # GraphSAGE, link predictor
│   │   ├── nlp.py               # FinBERT classifier
│   │   ├── lstm.py              # LSTM, TFT
│   │   ├── cnn.py               # ResNet regression
│   │   └── embeddings.py        # Sentence-BERT wrapper
│   ├── datasets/                # Data loaders (OEC, FOMC, WDI, VIIRS, ESCO)
│   └── utils/                   # Metrics, I/O, visualisation
├── scripts/                     # Data download helpers
├── tests/                       # Unit tests (models, datasets, metrics)
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Hardware Requirements

| Experiment | Min VRAM | CPU-only? | Approx. train time |
|---|---|---|---|
| Economic Complexity GNN | 4 GB | ✅ (slow) | 2 h on RTX 3080 |
| Central Bank NLP | 8 GB | ✅ (slow) | 45 min on RTX 3080 |
| GDP Nowcasting | 4 GB | ✅ | 30 min on CPU |
| Informal Economy | 8 GB | ✅ (slow) | 3 h on RTX 3080 |
| Labour Market Mismatch | 4 GB | ✅ | 20 min on CPU |

---

## Key References

1. Hidalgo & Hausmann (2009). [The building blocks of economic complexity](https://doi.org/10.1073/pnas.0900943106). *PNAS.*
2. Hamilton, Ying & Leskovec (2017). [Inductive representation learning on large graphs](https://arxiv.org/abs/1706.02216). *NeurIPS.*
3. Araci (2019). [FinBERT: Financial sentiment analysis with pre-trained language models](https://arxiv.org/abs/1908.10063).
4. Lim et al. (2021). [Temporal Fusion Transformers for multi-horizon forecasting](https://doi.org/10.1016/j.ijforecast.2021.03.012). *IJF.*
5. Henderson, Storeygard & Weil (2012). [Measuring economic growth from outer space](https://doi.org/10.1257/aer.102.2.994). *AER.*
6. Reimers & Gurevych (2019). [Sentence-BERT](https://arxiv.org/abs/1908.10084). *EMNLP.*
7. IMF (2022). [Nowcasting GDP using machine learning](https://www.imf.org/en/Publications/WP/Issues/2022/02/04/Nowcasting-GDP-513150). WP/22/26.

---

## Contributing

Issues and pull requests are welcome. If you use any of these experiments in research, a citation or acknowledgement is appreciated but not required.
