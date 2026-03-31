# Economics ML Portfolio

**Machine learning applied to development economics, macroeconomics, and labour markets.**

This portfolio applies deep learning to problems that matter to economists and policy makers but are underserved by open-source tooling: decoding central bank communication, measuring the informal economy from space, nowcasting GDP in data-scarce environments, and mapping the structure of international trade.

Every experiment uses freely available public data, is fully reproducible via Docker, and ships with a detailed per-experiment README covering methodology, results, and limitations.

---

## Experiments

### ✅ Completed

| # | Experiment | Task | Architecture | Key result |
|---|---|---|---|---|
| 1 | [Economic Complexity GNN](experiments/economic-complexity/) | Predict ECI + future export products | GraphSAGE + link predictor | NDCG@10 = 0.71 |
| 2 | [Central Bank NLP](experiments/central-bank-nlp/) | Classify FOMC/ECB minutes hawkish/neutral/dovish | FinBERT fine-tune | Macro-F1 = 0.83 |
| 3 | [GDP Nowcasting](experiments/gdp-nowcasting/) | Nowcast quarterly GDP with alt-data | Temporal Fusion Transformer | MAE = 0.81 pp |

### 🚧 Works in Progress

| # | Experiment | Task | Architecture | Status |
|---|---|---|---|---|
| 4 | [Informal Economy Detection](experiments/informal-economy/) | Estimate informal activity from VIIRS satellite imagery | ResNet-18 regression | Data pipeline ✅, training in progress |
| 5 | [Labour Market Mismatch](experiments/labor-market-mismatch/) | Map skills supply/demand gaps from job postings | Sentence-BERT + UMAP | Embeddings ✅, clustering in progress |

---

## Why these problems?

**Economic Complexity** — The product space (Hidalgo & Hausmann 2009) is one of the most influential ideas in development economics. It models world trade as a bipartite graph connecting countries to the products they export competitively. Predicting where a country moves in this network has direct implications for industrial strategy — the World Bank and IMF use ECI as a development indicator. Almost no open-source GNN implementations exist on this data.

**GDP Nowcasting** — In low-income countries, official GDP statistics are released 6–18 months late and revised substantially. High-frequency alternative signals (nighttime satellite light, Google Trends, mobile activity) can fill this gap. The IMF's Statistics Department actively works on this; open reproducible baselines help researchers everywhere.

**Central Bank NLP** — Monetary policy communication moves bond markets, FX rates, and inflation expectations. The current open-source standard is the Loughran-McDonald financial word list. Fine-tuned language models consistently outperform it, but no clean multi-bank public implementation exists. This experiment covers FOMC, ECB, and BOE.

**Informal Economy Detection** — The informal sector accounts for 30–60 % of GDP in many developing countries yet is largely invisible in national accounts. VIIRS nighttime luminosity provides a rare external proxy. This is an active frontier in development economics (Henderson, Storeygard & Weil 2012; World Bank 2021).

**Labour Market Mismatch** — Structural unemployment is partly driven by skills mismatch: workers supply skills that employers do not demand, and vice versa. Embedding job descriptions and skills taxonomies in a shared semantic space lets you quantify this mismatch at scale — directly useful for education policy and reskilling programmes.

---

## Running an experiment

All experiments share a single entry point. Pass the path to the experiment's YAML config:

```bash
# From the repository root
python src/main.py --config experiments/central-bank-nlp/config.yaml
```

### Smoke test (no GPU, no data download needed)

```bash
python src/main.py --config experiments/central-bank-nlp/config.yaml --smoke
```

### Docker (recommended for full runs)

```bash
docker build -t econ-ml:latest .

docker run -it --gpus all \
  -v /your/data:/data \
  econ-ml:latest \
  --config experiments/economic-complexity/config.yaml
```

### Data download

Each experiment has a download helper that fetches data from free public sources:

```bash
python scripts/download_fomc.py       --out data/fomc
python scripts/download_oec.py        --out data/oec
python scripts/download_wdi.py        --out data/wdi
python scripts/download_nightlights.py --out data/nightlights
python scripts/download_esco.py       --out data/esco
```

---

## Datasets

| Dataset | Source | Size | Licence |
|---|---|---|---|
| OEC / BACI trade flows | [oec.world](https://oec.world) / [CEPII](http://www.cepii.fr) | ~2 GB | CC-BY |
| World Bank WDI | [data.worldbank.org](https://data.worldbank.org) | ~300 MB | CC-BY 4.0 |
| FOMC minutes & speeches | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm) | ~50 MB | Public domain |
| ECB speeches | [ecb.europa.eu](https://www.ecb.europa.eu/press/key/html/index.en.html) | ~30 MB | ECB open data |
| BOE MPC minutes | [bankofengland.co.uk](https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes) | ~20 MB | Open Government Licence |
| VIIRS annual composites | [eogdata.mines.edu](https://eogdata.mines.edu/products/vnl/) | ~4 GB | Public domain |
| ESCO skills taxonomy v1.2 | [esco.ec.europa.eu](https://esco.ec.europa.eu/en/use-esco/download) | ~20 MB | CC-BY 4.0 |
| Google Trends (via pytrends) | [trends.google.com](https://trends.google.com) | on demand | Terms of Service |

---

## Repository structure

```
econ-ml-portfolio/
├── experiments/
│   ├── economic-complexity/
│   │   ├── README.md              methodology, results, limitations
│   │   ├── config.yaml            experiment configuration
│   │   └── config_smoke.yaml      tiny config for CI smoke test
│   ├── gdp-nowcasting/
│   ├── central-bank-nlp/
│   ├── informal-economy/
│   └── labor-market-mismatch/
├── src/
│   ├── main.py                    single entry point
│   ├── trainer.py                 dispatcher → experiment runners
│   ├── experiments/               per-experiment training logic
│   │   ├── economic_complexity.py
│   │   ├── gdp_nowcasting.py
│   │   ├── central_bank_nlp.py
│   │   ├── informal_economy.py
│   │   └── labor_market.py
│   ├── models/                    model architectures
│   │   ├── gnn.py                 GraphSAGE, link predictor
│   │   ├── nlp.py                 FinBERT classifier, SentimentIndex
│   │   ├── lstm.py                LSTM, Temporal Fusion Transformer
│   │   ├── cnn.py                 ResNet regression for satellite tiles
│   │   └── embeddings.py          Sentence-BERT wrapper, skill embedder
│   ├── datasets/                  dataset loaders
│   │   ├── oec.py
│   │   ├── fomc.py
│   │   ├── wdi.py
│   │   ├── nightlights.py
│   │   └── esco.py
│   └── utils/
│       ├── metrics.py             NDCG, MAE, F1, correlation utilities
│       ├── io.py                  config loading, checkpoint helpers
│       └── viz.py                 matplotlib plotting helpers
├── scripts/
│   ├── download_fomc.py
│   ├── download_oec.py
│   ├── download_wdi.py
│   ├── download_nightlights.py
│   └── download_esco.py
├── tests/
│   ├── test_models.py
│   ├── test_datasets.py
│   └── test_metrics.py
├── images/
│   └── banner.svg
├── .github/workflows/ci.yml
├── Dockerfile
├── docker_entrypoint.sh
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── .dockerignore
```

---

## Hardware requirements

| Experiment | Minimum VRAM | CPU-only? | Approx. train time |
|---|---|---|---|
| Economic Complexity GNN | 4 GB | ✅ (slow) | 2 h on RTX 3080 |
| Central Bank NLP | 8 GB | ✅ (slow) | 45 min on RTX 3080 |
| GDP Nowcasting | 4 GB | ✅ | 30 min on CPU |
| Informal Economy | 8 GB | ✅ (slow) | 3 h on RTX 3080 |
| Labour Market Mismatch | 4 GB | ✅ | 20 min on CPU |

---

## Relevant literature

1. Hidalgo, C. A., & Hausmann, R. (2009). [The building blocks of economic complexity](https://doi.org/10.1073/pnas.0900943106). *PNAS*.
2. Mealy, P., Farmer, J. D., & Teytelboym, A. (2019). [Interpreting economic complexity](https://doi.org/10.1126/sciadv.aau1705). *Science Advances*.
3. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). [Inductive representation learning on large graphs](https://arxiv.org/abs/1706.02216). *NeurIPS*.
4. Lim, B., et al. (2021). [Temporal Fusion Transformers for interpretable multi-horizon time series forecasting](https://doi.org/10.1016/j.ijforecast.2021.03.012). *International Journal of Forecasting*.
5. Araci, D. (2019). [FinBERT: Financial sentiment analysis with pre-trained language models](https://arxiv.org/abs/1908.10063).
6. Chen, X., & Nordhaus, W. D. (2011). [Using luminosity data as a proxy for economic statistics](https://doi.org/10.1073/pnas.1017031108). *PNAS*.
7. Henderson, J. V., Storeygard, A., & Weil, D. N. (2012). [Measuring economic growth from outer space](https://doi.org/10.1257/aer.102.2.994). *AER*.
8. Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence embeddings using Siamese BERT-networks](https://arxiv.org/abs/1908.10084). *EMNLP*.
9. IMF (2022). [Nowcasting GDP using machine learning: A discussion of recent developments](https://www.imf.org/en/Publications/WP/Issues/2022/02/04/Nowcasting-GDP-513150).

---

## Contributing

Issues and pull requests are welcome. If you use any of these experiments in research, a citation or acknowledgement is appreciated but not required.

