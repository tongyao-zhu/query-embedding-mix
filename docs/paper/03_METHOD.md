# Section 3: Method

The method section is implemented by the retrieval scripts in `query_embedding_mix/`.

## 3.1 Task And Data

- Dataset: mMARCO development queries and multilingual passage collections.
- Languages: `ar de en es fr hi id it ja nl pt ru vi zh`.
- Query pool: 1,484 sufficiently long aligned queries.
- Main corpus: full mMARCO, approximately 8.8M passages.
- Subset corpus: 100k passages per document-language setting for word-mix validation and ablations.
- Main retrieval model: `BAAI/bge-m3`.
- Retrieval: L2-normalized vectors, FAISS search, nDCG@10 as the primary metric.

Relevant files:

- `query_embedding_mix/download_mmarco_queries.py`
- `query_embedding_mix/encode_multilingual_corpus.py`
- `query_embedding_mix/evaluate.py`

## 3.2 Word-Level Mixing

Word-level code-mixed queries are used as a validation probe rather than the main controlled protocol.

Relevant files:

- `query_embedding_mix/generate_cm_bands.py`
- `query_embedding_mix/mix_count.py`
- `scripts/generate_word_mix.sh`
- `scripts/reproduce_word_mix.sh`

## 3.3 Embedding-Level Mixing

The main protocol constructs:

```text
e_mix(lambda) = normalize((1 - lambda) * e_L1 + lambda * e_L2)
```

with `lambda` in `{0, 10, 30, 50, 70, 90, 100}`.

Relevant files:

- `query_embedding_mix/onepass_dense_mix_run_custom_lang.py`
- `query_embedding_mix/onepass_bilingual_mix_hub_custom_lang.py`
- `query_embedding_mix/cache_queries_for_mix.py`

## 3.4 Language Pair Metadata

Metadata used for script, typology, family, and resource-level analysis is stored in:

- `configs/language_pairs_typology_metrics.csv`

## 3.5 Evaluation And Summary Measures

The paper compares the best interior ratio with the best endpoint:

```text
endpoint_best = max(metric(lambda=0), metric(lambda=100))
best_mid      = max(metric(lambda in {10, 30, 50, 70, 90}))
delta         = best_mid - endpoint_best
```

Relevant files:

- `query_embedding_mix/evaluate.py`
- `query_embedding_mix/collect_results.py`
- `query_embedding_mix/calculate_paper_values.py`
