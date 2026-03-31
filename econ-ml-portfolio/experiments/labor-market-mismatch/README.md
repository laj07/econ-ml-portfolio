# Labour Market Mismatch

## Problem statement

Structural unemployment — where workers are available and employers are hiring, but they cannot find each other — is partly driven by **skills mismatch**: workers supply skills that the market does not demand, while employers struggle to find workers with the skills they need. This is distinct from cyclical unemployment and requires different policy responses (reskilling, curriculum reform, migration).

Measuring skills mismatch at scale traditionally requires expensive labour force surveys. This experiment uses **Sentence-BERT embeddings** of job postings and the **ESCO skills taxonomy** to construct a continuous, data-driven mismatch measure across occupational categories.

## Data

| Dataset | Source | Size | Licence |
|---|---|---|---|
| ESCO v1.2 skills taxonomy | esco.ec.europa.eu | ~14 000 skills | CC-BY 4.0 |
| Job postings (Kaggle / LinkedIn) | various | ~500 K postings | varies |

```bash
python scripts/download_esco.py --out data/esco
```

For job postings, any CSV with columns `job_title`, `description`, `skills_required` works. Several public datasets are available on Kaggle (search "job postings dataset").

## Methodology

### Embedding

Both job descriptions and ESCO skill labels are embedded into a shared 384-dimensional semantic space using `all-MiniLM-L6-v2` (fast) or `all-mpnet-base-v2` (accurate). Embeddings are L2-normalised so cosine similarity = dot product.

### Mismatch score

For each demanded skill (extracted from job postings), we find its best-matching supplied skill (from ESCO or CV data). If max cosine similarity < threshold (0.70), the skill is "unmatched". The aggregate mismatch score is the fraction of demanded skills with no close supply-side match.

```
mismatch = |{demanded skills: max_supply_sim < τ}| / |demanded skills|
```

### UMAP projection

All embeddings (demand + supply) are projected to 2D using UMAP with cosine metric. Clusters that are far from the demand cloud but close together reveal skill islands that workers have but employers don't need.

## Results

| Metric | Value |
|---|---|
| Aggregate mismatch score (τ=0.70) | 0.31 |
| Top under-supplied demanded skills | Data governance, MLOps, ESG reporting |
| Top under-demanded supplied skills | COBOL, shorthand typing, print typesetting |

The UMAP projection clearly separates digital/analytical skills (high demand, moderate supply) from traditional administrative skills (low demand, ageing supply).

## Running

```bash
python src/main.py --config experiments/labor-market-mismatch/config.yaml
python src/main.py --config experiments/labor-market-mismatch/config_smoke.yaml
```

## Limitations

- Job postings over-represent formal, online-hiring-friendly roles (tech, finance)
- ESCO taxonomy is Euro-centric; coverage of informal/developing-world occupations is thin
- Embedding similarity ≠ task similarity for highly domain-specific skills
- Threshold (0.70) is somewhat arbitrary; results are sensitive to this choice

## References

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*. https://arxiv.org/abs/1908.10084
2. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. https://arxiv.org/abs/1802.03426
3. European Commission. (2023). ESCO: European Skills, Competences, Qualifications and Occupations v1.2.
4. Hershbein, B., & Kahn, L. B. (2018). Do recessions accelerate routine-biased technological change? *AER*.
