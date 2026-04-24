# Appendix Workflows

This folder is for workflows that support the paper but are not the main full-corpus embedding-mix experiment.

## Word-Mix Validation

The paper uses word-level code-mixed queries to check whether embedding-level interpolation follows similar ratio trends. This validation covers:

```text
EN-ZH, EN-VI, ZH-VI, HI-ID
```

Generate word-mixed query bands:

```bash
cp .env.example .env
# Fill OPENROUTER_API_KEY or OPENAI_API_KEY.

bash scripts/generate_word_mix.sh
```

Run word-mix retrieval and the matching embedding-mix comparison:

```bash
bash scripts/reproduce_word_mix.sh
```

Checked-in summaries:

- `artifacts/tables/word_mix_curves.csv`
- `artifacts/tables/word_mix_processed.csv`

## 100k Subset

Word-mix validation and ablations use 100k-passage subsets to reduce compute while keeping judged-relevant passages retrievable. The relevant indexing code is:

- `query_embedding_mix/encode_multilingual_corpus.py`
- `scripts/reproduce_word_mix.sh`
- `scripts/run_encode_index_ablation.sh`

## Paper-Only Side Analyses

The LaTeX source also discusses exploratory side analyses such as strict non-Latin purity checks and a lightweight router. These are documented in the paper map but are intentionally not part of the core public reproduction path.

See [../paper/APPENDICES.md](../paper/APPENDICES.md) for the full appendix-to-repo map.
