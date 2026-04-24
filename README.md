# Query Embedding Mix

Code and compact artifacts for the ACL 2026 paper:
`When Does Mixing Help? Analyzing Query Embedding Interpolation in Multilingual Dense Retrieval`

This repository is the cleaned paper-facing version of the earlier experimental workspace. It keeps the core reproduction pipelines, checked-in summary tables, selected example outputs, and final paper figures, while intentionally omitting side tools that are not part of the paper’s main reproducibility story.

## What Is Included

- Main vector-mix experiments on mMARCO
- Word-mix validation for `EN-ZH`, `EN-VI`, `ZH-VI`, and `HI-ID`
- Model-family and scale ablations
- EN-ZH embedding-space analysis
- Final compact tables and paper figures

## What Is Intentionally Omitted

These were kept out of this paper repo to reduce clutter:

- `micro_case_tool/`
- `mmarco_simple_router/`
- the non-Latin purity appendix workflow
- large raw `indexes/`, `runs/`, `results/`, and `logs/` trees

## Repository Layout

```text
query-embedding-mix/
├── query_embedding_mix/   # core Python code
├── scripts/               # experiment entrypoints
├── configs/               # small checked-in metadata / qid filters
├── artifacts/
│   ├── tables/            # compact summary CSVs used in the paper
│   ├── analysis/          # checked-in EN-ZH embedding-space outputs
│   └── examples/          # small example result bundles
├── assets/figures/        # final paper figures
├── requirements.txt
└── README.md
```

Generated runtime directories are still expected under the repo root when you run experiments:

- `data/`
- `indexes/`
- `runs/`
- `results/`
- `logs/`

## Setup

```bash
conda create -n query-mix python=3.11.13 -y
conda activate query-mix
conda install -c conda-forge faiss-gpu=1.8.0 -y
pip install -r requirements.txt
```

For word-mix generation, install the Stanza tokenizers you need:

```bash
python - <<'PY'
import stanza
for lang in ["en", "zh", "vi", "hi", "id"]:
    stanza.download(lang)
PY
```

## Main Reproduction

### 1) Download mMARCO queries

```bash
python query_embedding_mix/download_mmarco_queries.py \
  --out_dir ./data/mmarco_dev \
  --split dev \
  --languages english chinese french german italian spanish portuguese dutch russian japanese arabic hindi indonesian vietnamese
```

### 2) Build the BGE-M3 indexes

```bash
bash scripts/run_encode_index_groups.sh
```

### 3) Run all vector-mix experiments

```bash
INDEX_ROOT=./indexes/idx-mmarco-bge-m3-sub8841823 \
bash scripts/run_all_vector_pairs.sh
```

### 4) Collect the main tables

```bash
python query_embedding_mix/collect_results.py \
  ./results/mmarco_full \
  --output ./artifacts/tables/full_mmarco_results.csv \
  --processed-out ./artifacts/tables/full_mmarco_processed_results.csv
```

## Word-Mix Validation

The word-mix appendix workflow uses four pairs:

- `EN-ZH`
- `EN-VI`
- `ZH-VI`
- `HI-ID`

Generate code-mixed query bands:

```bash
bash scripts/generate_word_mix.sh
```

Then reproduce retrieval with both word-mix and embedding-mix:

```bash
bash scripts/reproduce_word_mix.sh
```

Checked-in compact summaries for this appendix workflow are available at:

- `artifacts/tables/word_mix_curves.csv`
- `artifacts/tables/word_mix_processed.csv`

The shared qid filter used by default is checked in at:

- `configs/qid_lists/en_zh_min.ge6.tsv`

## Ablations

Build ablation indexes:

```bash
bash scripts/run_encode_index_ablation.sh
```

Run the ablation matrix:

```bash
bash scripts/run_ablation.sh
```

Collect the ablation tables:

```bash
python query_embedding_mix/collect_ablation_results.py \
  ./results/mmarco_full/ablation2 \
  --output ./artifacts/tables/ablation_results.csv \
  --processed-out ./artifacts/tables/ablation_processed_results.csv
```

## EN-ZH Embedding-Space Analysis

This analysis is only for the `EN-ZH` pair. It is intentionally named explicitly in this repo.

Run:

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

## Checked-In Artifacts

- `artifacts/tables/`: compact paper-facing CSV summaries
- `artifacts/tables/word_mix_*.csv`: compact word-mix appendix summaries
- `artifacts/analysis/en_zh_embedding_space/`: checked-in EN-ZH geometry outputs
- `artifacts/examples/repro_en_zh_example/`: small example result bundle
- `assets/figures/`: final figures used in the paper

To regenerate the figures from the checked-in tables:

```bash
python query_embedding_mix/plot_paper_figures.py
```

To recompute the printed paper statistics from the checked-in tables:

```bash
python query_embedding_mix/calculate_paper_values.py
```
