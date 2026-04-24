# Script Map

The repo uses direct scripts rather than a large experiment framework. This page maps each entrypoint to its role.

## Shell Entrypoints

| Script | Role |
| --- | --- |
| `scripts/run_encode_index_groups.sh` | Build full BGE-M3 mMARCO indexes for all 14 languages. |
| `scripts/run_all_vector_pairs.sh` | Run the main 35-pair vector interpolation matrix. |
| `scripts/generate_word_mix.sh` | Generate LLM-based word-level code-mixed query bands. |
| `scripts/reproduce_word_mix.sh` | Reproduce word-mix validation and matching embedding-mix runs on 100k subsets. |
| `scripts/run_encode_index_ablation.sh` | Build indexes for model-family and scale ablations. |
| `scripts/run_ablation.sh` | Run ablation retrieval jobs. |

## Python Entrypoints

| Script | Role |
| --- | --- |
| `query_embedding_mix/download_mmarco_queries.py` | Download aligned mMARCO query translations into local TSV files. |
| `query_embedding_mix/encode_multilingual_corpus.py` | Encode multilingual document collections and write FAISS indexes. |
| `query_embedding_mix/onepass_dense_mix_run_custom_lang.py` | Run monolingual-document retrieval with interpolated query embeddings. |
| `query_embedding_mix/onepass_bilingual_mix_hub_custom_lang.py` | Run bilingual-document retrieval with interpolated query embeddings. |
| `query_embedding_mix/onepass_dense_run.py` | Run standard dense retrieval, including word-mix query files. |
| `query_embedding_mix/onepass_bilingual_hub.py` | Run standard retrieval over bilingual document indexes. |
| `query_embedding_mix/cache_queries_for_mix.py` | Cache monolingual query embeddings reused by interpolation jobs. |
| `query_embedding_mix/generate_cm_bands.py` | Generate ratio-bucketed word-level code-mixed query TSVs. |
| `query_embedding_mix/mix_count.py` | Count source/target token shares for code-mix validation. |
| `query_embedding_mix/evaluate.py` | Compute retrieval metrics from run files and qrels. |
| `query_embedding_mix/collect_results.py` | Collect and process main experiment results. |
| `query_embedding_mix/collect_ablation_results.py` | Collect and process ablation results. |
| `query_embedding_mix/en_zh_embedding_space_analysis.py` | Analyze EN-ZH word-mix embedding geometry. |
| `query_embedding_mix/plot_paper_figures.py` | Generate paper figures from compact tables. |
| `query_embedding_mix/calculate_paper_values.py` | Print paper-facing numeric summaries from checked-in tables. |

## Common Environment Variables

| Variable | Used By | Meaning |
| --- | --- | --- |
| `PYTHON_BIN` | Shell scripts | Python executable to use. |
| `DATA_ROOT` | Most scripts | Root for downloaded queries and generated code-mix files. |
| `INDEX_ROOT` | Retrieval scripts | Existing FAISS index root. |
| `INDEX_ROOT_BASE` | Indexing scripts | Parent directory for newly built indexes. |
| `RUN_ROOT` | Retrieval scripts | TREC run output root. |
| `RESULT_ROOT` | Retrieval scripts | Evaluation output root. |
| `LOG_DIR` | Shell scripts | Log directory. |
| `GPUS` | Main vector script | Space-separated GPU IDs for scheduling. |
| `QUERY_CACHE_ROOT` | Mix scripts | Cached monolingual query embeddings. |
| `QID_LIST` | Word-mix generation | Shared qid filter. |
| `OPENROUTER_API_KEY` or `OPENAI_API_KEY` | Word-mix generation | API key for OpenAI-compatible generation. |

## Notes for Extending Scripts

- Keep defaults close to the paper setup.
- Make hardware and path assumptions configurable through environment variables.
- Write compact summaries under `artifacts/tables/`.
- Keep large raw outputs under ignored runtime directories.
