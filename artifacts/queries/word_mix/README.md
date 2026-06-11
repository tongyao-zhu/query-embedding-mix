# Word-Mix Query Bundles

This directory holds checked-in query bundles that are small enough to review and reuse in the appendix workflows.

Current contents:

- `en-zh/`: checked-in EN-ZH word-mix bundle.
- `en-vi/`: checked-in EN-VI word-mix bundle.
- `zh-vi/`: checked-in ZH-VI word-mix bundle.
- `hi-id/`: checked-in HI-ID word-mix bundle.

The bundle directories are copied from previously generated query-band outputs. Use each pair's `qids-common.tsv` as the shared evaluation subset when comparing across bands.

Run all checked-in bundles:

```bash
CM_DIR_EN_ZH=./artifacts/queries/word_mix/en-zh \
COMMON_QIDS_EN_ZH=./artifacts/queries/word_mix/en-zh/qids-common.tsv \
CM_DIR_EN_VI=./artifacts/queries/word_mix/en-vi \
COMMON_QIDS_EN_VI=./artifacts/queries/word_mix/en-vi/qids-common.tsv \
CM_DIR_ZH_VI=./artifacts/queries/word_mix/zh-vi \
COMMON_QIDS_ZH_VI=./artifacts/queries/word_mix/zh-vi/qids-common.tsv \
CM_DIR_HI_ID=./artifacts/queries/word_mix/hi-id \
COMMON_QIDS_HI_ID=./artifacts/queries/word_mix/hi-id/qids-common.tsv \
bash scripts/reproduce_word_mix.sh
```

Run one checked-in pair:

```bash
ACTIVE_PAIRS_ENV="zh-vi" \
CM_DIR_ZH_VI=./artifacts/queries/word_mix/zh-vi \
COMMON_QIDS_ZH_VI=./artifacts/queries/word_mix/zh-vi/qids-common.tsv \
bash scripts/reproduce_word_mix.sh
```

Regenerate all four appendix pairs instead:

```bash
cp .env.example .env
# Fill OPENROUTER_API_KEY or OPENAI_API_KEY.

bash scripts/generate_word_mix.sh
bash scripts/reproduce_word_mix.sh
```
