# Section 4: Results

The results section is primarily backed by the full mMARCO vector-mix pipeline and the checked-in compact tables.

## 4.1 Comparing Word And Embedding Mixing

Purpose: validate that embedding-level interpolation tracks the ratio trends observed under generated word-level code mixing.

The LaTeX source uses `fig:enzh_proxy` for EN-ZH word-vs-embedding curves and `fig:embed_panels_app` for the EN-ZH embedding-space diagnostics.

Relevant files:

- `scripts/generate_word_mix.sh`
- `scripts/reproduce_word_mix.sh`
- `artifacts/tables/word_mix_curves.csv`
- `artifacts/tables/word_mix_processed.csv`
- `artifacts/analysis/en_zh_embedding_space/`

## 4.2 Global Picture: Embedding Interpolation

Paper claim: interior mixtures outperform the better monolingual endpoint in 88/105 main BGE-M3 settings.

The paper reports mean delta `+0.70`, median delta `+0.65`, and range `-0.34` to `+2.92` over the 105 paper settings.

Relevant files:

- `scripts/run_encode_index_groups.sh`
- `scripts/run_all_vector_pairs.sh`
- `query_embedding_mix/collect_results.py`
- `artifacts/tables/full_mmarco_results.csv`
- `artifacts/tables/full_mmarco_processed_results.csv`
- `assets/figures/delta_distribution_all.png`
- `assets/figures/lambda_star_summary.png`

## 4.3 The Strong Effects Of English

Paper claim: English is asymmetric. Mixing helps broadly when English is absent from the index, while English-containing indexes often prefer pure English.

The paper reports that non-English document settings have positive point-estimate deltas, while English-containing document settings cluster near zero or below.

Relevant figures:

- `assets/figures/en_in_index_split.png`
- `assets/figures/hub_sweeps.png`

## 4.4 Document-Query Language Match

Paper claim: monolingual document indexes still prefer the endpoint matching the document language, but interior mixtures often improve over the best endpoint.

Relevant file:

- `query_embedding_mix/calculate_paper_values.py`

## 4.5 Other Findings

Paper claim: after controlling for English dominance, larger typological distance correlates with smaller mixing gains.

Relevant files:

- `configs/language_pairs_typology_metrics.csv`
- `assets/figures/typology_scatter.png`
- `assets/figures/headroom_scatter.png`
- `assets/figures/mono_alignment_curve.png`

## Scope Note

The paper reports the representative 35-pair/105-setting subset. The checked-in `full_mmarco_processed_results.csv` currently includes an expanded 91-pair/273-setting grid for broader inspection.
