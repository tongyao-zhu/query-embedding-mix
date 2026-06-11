# Python Modules By Paper Component

| Paper component | Modules |
| --- | --- |
| Section 3.1 task and data | `download_mmarco_queries.py`, `encode_multilingual_corpus.py` |
| Section 3.2 word-level mixing | `generate_cm_bands.py`, `mix_count.py` |
| Section 3.3 embedding-level mixing | `onepass_dense_mix_run_custom_lang.py`, `onepass_bilingual_mix_hub_custom_lang.py`, `cache_queries_for_mix.py` |
| Standard retrieval baselines | `onepass_dense_run.py`, `onepass_bilingual_hub.py` |
| Section 3.5 evaluation | `evaluate.py`, `collect_results.py`, `collect_ablation_results.py` |
| Section 4.1 geometry diagnostics | `en_zh_embedding_space_analysis.py` |
| Paper figures and values | `plot_paper_figures.py`, `calculate_paper_values.py` |

The modules are mostly CLI scripts because this repository is organized for paper reproduction rather than as an import-first library.
