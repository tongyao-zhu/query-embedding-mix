# Code-Mixed Embedding Space Analysis Report

- Model: `BAAI/bge-m3`
- ABTT removed PCs: `0`
- Total aligned rows: `7420` across bands: `0-20, 20-40, 40-60, 60-80, 80-100`
- qids-common file: `data/mmarco_dev/queries_cm_5_bands_5-mini/qids-common.tsv` (|qids|=1484)

## Per-band summaries
## Outlier detector configuration
- `z_delta` MAD threshold: `3`
- `min_cos` percentile: `5.0%` → threshold = `0.6508`
- `r` margin outside [0,1]: `0.25`

## BEFORE outlier removal (all rows)
- **Row counts by band (rows = qid×band):**
  - `0-20`: rows=`1484`, unique qids=`1484`
  - `20-40`: rows=`1484`, unique qids=`1484`
  - `40-60`: rows=`1484`, unique qids=`1484`
  - `60-80`: rows=`1484`, unique qids=`1484`
  - `80-100`: rows=`1484`, unique qids=`1484`

### Per-band summaries (BEFORE)
#### Band `0-20`  (n=1484)
- mean r: `0.2849` | trimmed mean r: `0.2666` | median r: `0.1995` | frac r∈[0,1]: `0.993`
- mean δ: `0.2711` | trimmed mean δ: `0.2662` | median δ: `0.2587`
- mean δ_rel (δ/|EN–ZH|): `0.4454` | median δ_rel: `0.4431`
- mean α: `0.7151` | mean residual: `0.2711` | mean local R²: `0.6587`
#### Band `20-40`  (n=1484)
- mean r: `0.4255` | trimmed mean r: `0.4189` | median r: `0.3831` | frac r∈[0,1]: `0.997`
- mean δ: `0.2998` | trimmed mean δ: `0.2943` | median δ: `0.2827`
- mean δ_rel (δ/|EN–ZH|): `0.4958` | median δ_rel: `0.4773`
- mean α: `0.5745` | mean residual: `0.2998` | mean local R²: `0.5171`
#### Band `40-60`  (n=1484)
- mean r: `0.5683` | trimmed mean r: `0.5730` | median r: `0.5774` | frac r∈[0,1]: `0.996`
- mean δ: `0.3175` | trimmed mean δ: `0.3110` | median δ: `0.2974`
- mean δ_rel (δ/|EN–ZH|): `0.5284` | median δ_rel: `0.4967`
- mean α: `0.4317` | mean residual: `0.3175` | mean local R²: `0.3630`
#### Band `60-80`  (n=1484)
- mean r: `0.6789` | trimmed mean r: `0.6923` | median r: `0.7461` | frac r∈[0,1]: `0.982`
- mean δ: `0.3028` | trimmed mean δ: `0.2959` | median δ: `0.2883`
- mean δ_rel (δ/|EN–ZH|): `0.5032` | median δ_rel: `0.4795`
- mean α: `0.3211` | mean residual: `0.3028` | mean local R²: `0.2633`
#### Band `80-100`  (n=1484)
- mean r: `0.8015` | trimmed mean r: `0.8249` | median r: `0.8945` | frac r∈[0,1]: `0.956`
- mean δ: `0.2639` | trimmed mean δ: `0.2598` | median δ: `0.2511`
- mean δ_rel (δ/|EN–ZH|): `0.4413` | median δ_rel: `0.4157`
- mean α: `0.1985` | mean residual: `0.2639` | mean local R²: `0.1811`

### Cosine similarity (CM vs EN/ZH) by band (BEFORE)
- **0-20**: cos(cm,en) mean/median p10–p90: `0.9249` / `0.9536` [0.8171–0.9892] | cos(cm,zh): `0.8387` / `0.8529` [0.7101–0.9436]
- **20-40**: cos(cm,en) mean/median p10–p90: `0.8957` / `0.9183` [0.7833–0.9791] | cos(cm,zh): `0.8634` / `0.8839` [0.7378–0.9615]
- **40-60**: cos(cm,en) mean/median p10–p90: `0.8589` / `0.8778` [0.7299–0.9604] | cos(cm,zh): `0.8876` / `0.9121` [0.7625–0.9758]
- **60-80**: cos(cm,en) mean/median p10–p90: `0.8386` / `0.8524` [0.7145–0.9452] | cos(cm,zh): `0.9087` / `0.9385` [0.7856–0.9850]
- **80-100**: cos(cm,en) mean/median p10–p90: `0.8171` / `0.8281` [0.6939–0.9238] | cos(cm,zh): `0.9358` / `0.9637` [0.8352–0.9923]

## Outlier detection results
- Total outlier rows: `562` of `7420`; unique qids: `289`
- Outliers by band:
  - `0-20`: outlier rows=`91`, outlier qids=`91`
  - `20-40`: outlier rows=`94`, outlier qids=`94`
  - `40-60`: outlier rows=`122`, outlier qids=`122`
  - `60-80`: outlier rows=`122`, outlier qids=`122`
  - `80-100`: outlier rows=`133`, outlier qids=`133`
- Sample outlier qids (≤20): `1000004, 1000959, 1003277, 1003482, 1005949, 1006199, 1006489, 1006509, 1008947, 1010059, 1010277, 1012329, 1014132, 1016565, 1016676, 1017204, 1017537, 1017605, 1018918, 1019470`

## AFTER outlier removal (qid-wise strict)
- **Row counts by band (after row-wise clean):**
  - `0-20`: rows=`1195`, unique qids=`1195`
  - `20-40`: rows=`1195`, unique qids=`1195`
  - `40-60`: rows=`1195`, unique qids=`1195`
  - `60-80`: rows=`1195`, unique qids=`1195`
  - `80-100`: rows=`1195`, unique qids=`1195`

### Per-band summaries (AFTER, row-wise)
#### Band `0-20`  (n=1195)
- mean r: `0.2768` | trimmed mean r: `0.2589` | median r: `0.1940` | frac r∈[0,1]: `0.992`
- mean δ: `0.2624` | trimmed mean δ: `0.2589` | median δ: `0.2523`
- mean δ_rel (δ/|EN–ZH|): `0.4307` | median δ_rel: `0.4404`
- mean α: `0.7232` | mean residual: `0.2624` | mean local R²: `0.6748`
#### Band `20-40`  (n=1195)
- mean r: `0.4251` | trimmed mean r: `0.4185` | median r: `0.3812` | frac r∈[0,1]: `0.997`
- mean δ: `0.2913` | trimmed mean δ: `0.2875` | median δ: `0.2772`
- mean δ_rel (δ/|EN–ZH|): `0.4802` | median δ_rel: `0.4743`
- mean α: `0.5749` | mean residual: `0.2913` | mean local R²: `0.5234`
#### Band `40-60`  (n=1195)
- mean r: `0.5700` | trimmed mean r: `0.5748` | median r: `0.5815` | frac r∈[0,1]: `0.997`
- mean δ: `0.3045` | trimmed mean δ: `0.3003` | median δ: `0.2901`
- mean δ_rel (δ/|EN–ZH|): `0.5033` | median δ_rel: `0.4939`
- mean α: `0.4300` | mean residual: `0.3045` | mean local R²: `0.3707`
#### Band `60-80`  (n=1195)
- mean r: `0.6810` | trimmed mean r: `0.6943` | median r: `0.7443` | frac r∈[0,1]: `0.989`
- mean δ: `0.2896` | trimmed mean δ: `0.2862` | median δ: `0.2825`
- mean δ_rel (δ/|EN–ZH|): `0.4794` | median δ_rel: `0.4731`
- mean α: `0.3190` | mean residual: `0.2896` | mean local R²: `0.2661`
#### Band `80-100`  (n=1195)
- mean r: `0.8036` | trimmed mean r: `0.8248` | median r: `0.8908` | frac r∈[0,1]: `0.968`
- mean δ: `0.2516` | trimmed mean δ: `0.2503` | median δ: `0.2453`
- mean δ_rel (δ/|EN–ZH|): `0.4132` | median δ_rel: `0.4106`
- mean α: `0.1964` | mean residual: `0.2516` | mean local R²: `0.1820`

### Band `0-20`  (n=1484)
- mean r: `0.2849` | trimmed mean r (5%): `0.2666` | median r: `0.1995` | frac r∈[0,1]: `0.993`
- mean δ: `0.2711` | trimmed mean δ (5%): `0.2662` | median δ: `0.2587`
- mean α: `0.7151` | mean residual: `0.2711` | mean local R²: `0.6587`

### Band `20-40`  (n=1484)
- mean r: `0.4255` | trimmed mean r (5%): `0.4189` | median r: `0.3831` | frac r∈[0,1]: `0.997`
- mean δ: `0.2998` | trimmed mean δ (5%): `0.2943` | median δ: `0.2827`
- mean α: `0.5745` | mean residual: `0.2998` | mean local R²: `0.5171`

### Band `40-60`  (n=1484)
- mean r: `0.5683` | trimmed mean r (5%): `0.5730` | median r: `0.5774` | frac r∈[0,1]: `0.996`
- mean δ: `0.3175` | trimmed mean δ (5%): `0.3110` | median δ: `0.2974`
- mean α: `0.4317` | mean residual: `0.3175` | mean local R²: `0.3630`

### Band `60-80`  (n=1484)
- mean r: `0.6789` | trimmed mean r (5%): `0.6923` | median r: `0.7461` | frac r∈[0,1]: `0.982`
- mean δ: `0.3028` | trimmed mean δ (5%): `0.2959` | median δ: `0.2883`
- mean α: `0.3211` | mean residual: `0.3028` | mean local R²: `0.2633`

### Band `80-100`  (n=1484)
- mean r: `0.8015` | trimmed mean r (5%): `0.8249` | median r: `0.8945` | frac r∈[0,1]: `0.956`
- mean δ: `0.2639` | trimmed mean δ (5%): `0.2598` | median δ: `0.2511`
- mean α: `0.1985` | mean residual: `0.2639` | mean local R²: `0.1811`

## Correlations across bands
- Spearman(mix_midpoint, mean r) ≈ `1.0000`
- Spearman(mix_midpoint, mean δ) ≈ `-0.1000`

## Diagnostics & Plots
- Hubness stats: `hubness_stats.json`
- Anisotropy: `anisotropy.json`
- UMAP: `viz_umap.png` and `viz_umap_by_band.png`
- t-SNE: `viz_tsne.png` and `viz_tsne_by_band.png`

## Notes
- EN (gray circle), ZH (gray triangle). CM is a square colored by band.
- Short gray line segments show EN→CM and CM→ZH per (qid, band).
- Use r/δ/α for quantitative claims; UMAP/t-SNE are for intuition.

---
Generated by en_zh_embedding_space_analysis.py
