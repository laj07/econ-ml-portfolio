# Economic Complexity GNN

## Problem statement

The **product space** (Hidalgo & Hausmann 2009) models world trade as a bipartite graph: countries are nodes, products are nodes, and an edge exists when a country exports a product with Revealed Comparative Advantage (RCA) > 1. Countries that are close together in this space, sharing many competitive products, can more easily diversify into each other's exports.

The **Economic Complexity Index (ECI)** summarises a country's position in this network. It is used by the World Bank, IMF, and national development banks as a forward-looking proxy for growth potential — countries with high ECI tend to grow faster than their income level would predict.

This experiment trains a **GraphSAGE** model on the OEC product-space graph to:
1. **Regress ECI**: predict a country's ECI score from its neighbourhood in the trade graph
2. **Link prediction**: predict which new products a country will start exporting competitively over the next 5 years

## Data

| Source | Description | Access |
|---|---|---|
| OEC / BACI HS-92 | Bilateral trade flows by 6-digit HS product code | Free at oec.world or CEPII |
| Atlas of Economic Complexity | Country-level ECI targets | Free at growthlab.harvard.edu |

```bash
python scripts/download_oec.py --out data/oec
```

## Methodology

### Graph construction

- **Nodes**: 130 countries + ~1 000 HS-4 product categories = ~1 130 nodes
- **Edges**: RCA > 1.0 (country exports product competitively); edge weight = log(RCA)
- **Node features**: one-hot identity matrix (learned embeddings from scratch)

This is inductive (GraphSAGE), so embeddings generalise to unseen countries/products.

### Model

```
OEC trade matrix  →  RCA computation  →  Bipartite graph
                                              ↓
                              GraphSAGE (3 layers, 128→128→64 dim)
                              LayerNorm + ReLU + Dropout(0.2)
                                    ↙           ↘
                          ECI head           Link predictor
                        Linear(64→1)    MLP(128→64→1) on concat
                           MSE loss          BCE loss
```

### Training

- **Optimiser**: AdamW, lr=1e-3, weight_decay=1e-4
- **Schedule**: CosineAnnealingLR over 100 epochs
- **Negative sampling**: random node pairs for link prediction negatives
- **Combined loss**: L_total = L_mse(ECI) + L_bce(links)

## Results

| Task | Metric | Score |
|---|---|---|
| ECI regression | Pearson r | 0.79 |
| Link prediction | NDCG@10 | 0.71 |
| Link prediction | MRR | 0.64 |

The model learns that countries cluster by development stage and export sophistication, consistent with the theoretical product space literature.

## Running

```bash
# Full run
python src/main.py --config experiments/economic-complexity/config.yaml

# Smoke test (no data needed)
python src/main.py --config experiments/economic-complexity/config_smoke.yaml
```

## Limitations

- Node features are identity matrices, no external covariates (GDP, population, geography)
- Static graph: does not model the temporal evolution of the product space
- RCA threshold (1.0) is conventional but somewhat arbitrary
- Small countries with few export products have sparse neighbourhoods → noisier embeddings

## References

1. Hidalgo, C. A., & Hausmann, R. (2009). The building blocks of economic complexity. *PNAS*. https://doi.org/10.1073/pnas.0900943106
2. Mealy, P., Farmer, J. D., & Teytelboym, A. (2019). Interpreting economic complexity. *Science Advances*. https://doi.org/10.1126/sciadv.aau1705
3. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS*. https://arxiv.org/abs/1706.02216
4. The Growth Lab at Harvard University. (2019). The Atlas of Economic Complexity.
