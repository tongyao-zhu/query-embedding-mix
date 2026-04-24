# Appendix Workflows

This page maps appendix sections in the paper source to repository files. The public repo focuses on the main reproducibility path and compact paper-facing artifacts; a few exploratory side tools are documented but intentionally not included as core workflows.

## Appendix A: Word-Mix Query Generation Details

Purpose: generate ratio-bucketed word-level code-mixed queries for validation.

Paper subsections:

- A.1 Pre-filtering of parallel queries
- A.2 Generation task and constraints
- A.3 LLM configuration
- A.4 Operational definition of ZH share
- A.5 Band targets and ratio control loop
- A.6 Filtering and band assignment
- A.7 Comparable query sets across bands
- A.8 Full generation prompt
- A.9 Word-Mix Generation Samples

Relevant files:

- `query_embedding_mix/generate_cm_bands.py`
- `query_embedding_mix/mix_count.py`
- `scripts/generate_word_mix.sh`
- `configs/qid_lists/en_zh_min.ge6.tsv`

Environment:

```bash
cp .env.example .env
# Fill OPENROUTER_API_KEY or OPENAI_API_KEY.
```

## Appendix B: Constructing A 100k mMARCO Subset

Purpose: keep ablations and word-mix validation computationally tractable while preserving judged-relevant passages.

Relevant files:

- `query_embedding_mix/encode_multilingual_corpus.py`
- `scripts/reproduce_word_mix.sh`
- `scripts/run_encode_index_ablation.sh`

## Appendix C: Additional Experiments

Included in the compact public repo:

- More pairs for word-mix vs embed-mix: EN-ZH, EN-VI, ZH-VI, HI-ID.
- Model-family and scale ablation summaries.
- Low-resource extension and additional summaries when represented in checked-in compact tables.

Relevant files:

- `artifacts/tables/word_mix_curves.csv`
- `artifacts/tables/word_mix_processed.csv`
- `artifacts/tables/ablation_processed_results.csv`
- `assets/figures/`

Paper-only or intentionally omitted from the core public reproduction path:

- Strictly monolingual/non-Latin purity appendix workflow.
- Lightweight router side experiment.

These were kept out of the main repo surface to avoid expanding the reproduction target beyond the paper's central embedding-mix study.

## Appendix D: Additional Findings

Additional finding tables are summarized through:

- `query_embedding_mix/calculate_paper_values.py`
- `artifacts/tables/paper_values.txt`

## Appendix E: Additional Related Work

This is paper text only; no repository workflow is attached.

## Appendix F: License Of Artifacts

Repository license and artifact policy:

- `LICENSE`
- `docs/ARTIFACTS.md`

## Appendix G: Use Of LLMs

LLM use is limited to word-level code-mix query generation in the validation workflow.

Relevant files:

- `query_embedding_mix/generate_cm_bands.py`
- `.env.example`
