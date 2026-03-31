# Informal Economy Detection via Satellite Imagery

## Problem statement

The informal economy — unregistered businesses, street markets, subsistence agriculture, day labour — accounts for an estimated 30–60% of GDP in developing countries (ILO 2018) but is nearly invisible in official statistics. National accounts measure what passes through formal registration and tax systems.

**Nighttime satellite imagery** provides a rare external signal: economic activity requires and generates light. The VIIRS (Visible Infrared Imaging Radiometer Suite) sensor produces annual cloud-free composites at ~500m resolution — fine enough to distinguish urban commercial districts from residential areas.

This experiment trains a **ResNet-18 regression model** on 256×256 VIIRS luminosity tiles to predict subnational log GDP per capita, then uses the residuals (actual − predicted) as a proxy for informal activity: high residuals suggest an area is more economically active than its luminosity implies.

## Data

| Dataset | Source | Resolution | Access |
|---|---|---|---|
| VIIRS VNL v2 annual composites | eogdata.mines.edu (NOAA) | ~500m | Public domain |
| World Bank subnational poverty data | data.worldbank.org | ADM2 | CC-BY 4.0 |
| OpenStreetMap settlement labels | openstreetmap.org | vector | ODbL |

```bash
python scripts/download_nightlights.py --out data/nightlights
```

Tiles are 256×256 pixels (~128×128 km) centered on subnational administrative units. Ground truth is log GDP per capita from World Bank subnational poverty assessments.

## Methodology

### Preprocessing

- Radiance values: log1p transform → standardise (μ=0.847, σ=1.203)
- Tiles with >50% cloud/water mask are excluded
- Data augmentation: random horizontal/vertical flip, random 90° rotation

### Model

```
VIIRS tile (1 × 256 × 256)
        ↓
ResNet-18 backbone (ImageNet weights, conv1 adapted to 1 channel)
        ↓
Global average pool → 512-dim features
        ↓
Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.15) → Linear(128→1)
        ↓
  ŷ = log GDP per capita
```

ImageNet pre-training transfers better than expected to single-band imagery — luminosity patterns (urban density, infrastructure grids) share structural features with natural image textures.

### Informality proxy

After training:
```
residual_i = log_gdp_actual_i − log_gdp_predicted_i
```

Positive residuals → area is richer than its luminosity suggests → may have more informal activity. This is validated against World Bank enterprise survey data (% of firms reporting informal competitors).

## Results

| Metric | Value |
|---|---|
| Val MAE (log units) | 0.43 |
| Val RMSE (log units) | 0.61 |
| Pearson r (log GDP) | 0.72 |
| Corr. residuals vs informality survey | 0.38 |

## Running

```bash
python src/main.py --config experiments/informal-economy/config.yaml
python src/main.py --config experiments/informal-economy/config_smoke.yaml
```

## Limitations

- Annual composites miss seasonal informal activity (harvest markets, festivals)
- Urban light pollution bleeds across tile boundaries
- Informality proxy is noisy — enterprise surveys are themselves imperfect
- Model trained on South/Southeast Asia; transfer to Sub-Saharan Africa untested

## References

1. Chen, X., & Nordhaus, W. D. (2011). Using luminosity data as a proxy for economic statistics. *PNAS*. https://doi.org/10.1073/pnas.1017031108
2. Henderson, J. V., Storeygard, A., & Weil, D. N. (2012). Measuring economic growth from outer space. *AER*. https://doi.org/10.1257/aer.102.2.994
3. Elvidge, C. D., et al. (2021). Annual time series of global VIIRS nighttime lights. *Remote Sensing*.
4. ILO (2018). Women and men in the informal economy: A statistical picture.
