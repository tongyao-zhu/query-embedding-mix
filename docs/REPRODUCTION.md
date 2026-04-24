# Reproduction Guide

This guide documents the commands used by the public repo. Full mMARCO runs require substantial GPU time and disk space. Plotting and table inspection can be done from the checked-in compact artifacts.

## 1. Environment

```bash
conda create -n query-mix python=3.11.13 -y
conda activate query-mix
conda install -c conda-forge faiss-gpu=1.8.0 -y
pip install -r requirements.txt
```

Or:

```bash
conda env create -f environment.yml
conda activate query-mix
```

## 2. Download mMARCO Queries

```bash
python query_embedding_mix/download_mmarco_queries.py \
  --out_dir ./data/mmarco_dev \
  --split dev \
  --languages english chinese french german italian spanish portuguese dutch russian japanese arabic hindi indonesian vietnamese
```

Expected output:

```text
data/mmarco_dev/
|-- queries.ar.tsv
|-- queries.de.tsv
|-- ...
`-- queries.zh.tsv
```

## 3. Main BGE-M3 Full mMARCO Runs

Build per-language FAISS indexes:

```bash
bash scripts/run_encode_index_groups.sh
```

Run the vector-mix matrix:

```bash
INDEX_ROOT=./indexes/idx-mmarco-bge-m3-sub8841823 \
bash scripts/run_all_vector_pairs.sh
```

Useful environment variables:

```bash
DATA_ROOT=./data
RUN_ROOT=./runs
RESULT_ROOT=./results/mmarco_full
LOG_DIR=./logs/vector_pairs
GPUS="0 1"
PAIR_MODE=default
MONO_MODE=default
```

Collect result tables:

```bash
python query_embedding_mix/collect_results.py \
  ./results/mmarco_full \
  --output ./artifacts/tables/full_mmarco_results.csv \
  --processed-out ./artifacts/tables/full_mmarco_processed_results.csv
```

## 4. Word-Mix Validation

Install Stanza tokenizers for the word-mix languages:

```bash
python - <<'PY'
import stanza
for lang in ["en", "zh", "vi", "hi", "id"]:
    stanza.download(lang)
PY
```

Prepare API credentials for `generate_cm_bands.py`. By default, the script reads `.env` and uses an OpenAI-compatible endpoint configured for OpenRouter. A safe template is provided in `.env.example`.

Generate word-mixed query bands:

```bash
bash scripts/generate_word_mix.sh
```

Run retrieval for word-mix and corresponding embedding-mix conditions:

```bash
bash scripts/reproduce_word_mix.sh
```

Collect or inspect the compact summaries:

```text
artifacts/tables/word_mix_curves.csv
artifacts/tables/word_mix_processed.csv
```

## 5. Ablations

Build ablation indexes:

```bash
bash scripts/run_encode_index_ablation.sh
```

Run ablation jobs:

```bash
bash scripts/run_ablation.sh
```

Collect ablation tables:

```bash
python query_embedding_mix/collect_ablation_results.py \
  ./results/mmarco_full/ablation2 \
  --output ./artifacts/tables/ablation_results.csv \
  --processed-out ./artifacts/tables/ablation_processed_results.csv
```

## 6. EN-ZH Embedding-Space Analysis

```bash
python query_embedding_mix/en_zh_embedding_space_analysis.py \
  --en_file data/mmarco_dev/queries.en.tsv \
  --zh_file data/mmarco_dev/queries.zh.tsv \
  --cm_files data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm0-20.tsv data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm20-40.tsv data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm40-60.tsv data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm60-80.tsv data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm80-100.tsv \
  --cm_labels 0-20 20-40 40-60 60-80 80-100 \
  --qids_common_file data/mmarco_dev/queries_cm_5_bands_5-mini/qids-common.tsv \
  --model_name BAAI/bge-m3 \
  --output_dir artifacts/analysis/en_zh_embedding_space
```

Checked-in outputs live under:

```text
artifacts/analysis/en_zh_embedding_space/
```

## 7. Figures and Paper Values

Regenerate figures from checked-in tables:

```bash
python query_embedding_mix/plot_paper_figures.py
```

Recompute paper-facing numeric summaries:

```bash
python query_embedding_mix/calculate_paper_values.py
```

Outputs are written to:

```text
assets/figures/
artifacts/tables/paper_values.txt
```

## 8. Lightweight Sanity Checks

For documentation or table-only changes, these checks are usually enough:

```bash
python -m compileall query_embedding_mix
python query_embedding_mix/calculate_paper_values.py
```

The full retrieval commands are the authoritative check for experiment changes.
