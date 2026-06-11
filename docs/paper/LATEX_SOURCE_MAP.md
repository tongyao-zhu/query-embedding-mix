# LaTeX Source Map

This repository map was cross-checked against the paper LaTeX source. It records the correspondence between paper sections, labels, figures, and repository files.

## Main Paper Files

| LaTeX source | Paper content | Repo documentation |
| --- | --- | --- |
| `latex/intro.tex` | Introduction and study protocol illustration | Root `README.md` abstract and hero figure |
| `latex/method.tex` | Section 3: Method | [03_METHOD.md](03_METHOD.md) |
| `latex/results.tex` | Section 4: Results | [04_RESULTS.md](04_RESULTS.md) |
| `latex/ablation.tex` | Section 5: Ablation | [05_ABLATION.md](05_ABLATION.md) |
| `latex/appendix.tex` | Appendices A-G | [APPENDICES.md](APPENDICES.md) |

## Section Labels

| Label | Paper heading | Repo files |
| --- | --- | --- |
| `subsec:data` | 3.1 Task and Data | `download_mmarco_queries.py`, `encode_multilingual_corpus.py` |
| `subsec:wordmix` | 3.2 Word-Level Mixing | `generate_cm_bands.py`, `mix_count.py`, `scripts/generate_word_mix.sh` |
| `subsec:embedmix` | 3.3 Embedding-Level Mixing | `onepass_dense_mix_run_custom_lang.py`, `onepass_bilingual_mix_hub_custom_lang.py` |
| `subsec:langmeta` | 3.4 Language Pair Metadata | `configs/language_pairs_typology_metrics.csv` |
| `subsec:eval` | 3.5 Evaluation and Summary Measures | `evaluate.py`, `collect_results.py`, `calculate_paper_values.py` |
| `sec:proxy_results` | 4.1 Comparing Word and Embedding Mixing | `artifacts/tables/word_mix_*`, `en_zh_embedding_space_analysis.py` |
| `subsec:global_picture` | 4.2 Global Picture | `full_mmarco_processed_results.csv`, `delta_distribution_all`, `triad_ENZH` |
| `subsec:en_boundary` | 4.3 Strong Effects of English | `en_in_index_split`, `hub_sweeps` |
| `subsec:hub` | English strongest partner paragraph | `hub_sweeps`, `calculate_paper_values.py` |
| `subsec:alignment_lambda` | 4.4 Document-Query Language Match | `mono_alignment_curve`, `lambda_star_summary` |
| `subsec:metrics`, `subsec:typology` | 4.5 Other Findings | `typology_scatter`, `headroom_scatter`, paper value summaries |
| `sec:ablation` | 5 Ablation: Model Family and Scale | `ablation_processed_results.csv`, ablation figures |
| `sec:generation_details` | Appendix A Word-mix Query Generation Details | `generate_cm_bands.py`, `mix_count.py` |
| `app:subset` | Appendix B Constructing a 100k mMARCO subset | `encode_multilingual_corpus.py` |
| `sec:wordmix_more_pairs_app` | Appendix C More pairs for word-mix vs embed-mix | `scripts/reproduce_word_mix.sh`, `word_mix_curves.csv` |
| `sec:low_resource_extension_app` | Appendix C.2 Extension to low-resource languages | Compact paper summaries when available |
| `sec:purity_filtered_app` | Appendix C.3 Strictly monolingual non-Latin evaluation | Paper-only side workflow, not core public reproduction |
| `sec:router_policy_app` | Appendix C.4 Lightweight router | Paper-only side workflow, not core public reproduction |

## Figure Labels

| Paper label | LaTeX figure file | Repo figure or artifact |
| --- | --- | --- |
| `fig:teaser` | `diagrams/teaser_diagram.pdf` | README protocol description |
| `fig:embed_mix` | `diagrams/embed-mix_diagram.pdf` | Section 3 method docs |
| `fig:word_mix` | `diagrams/word-mix_diagram.pdf` | Appendix workflow docs |
| `fig:enzh_proxy` | `rebuttal_ratio_curve_ENZH_*.pdf` | `artifacts/tables/word_mix_curves.csv` |
| `fig:embed_panels_app` | `r_position.pdf`, `delta_offset.pdf` | `assets/figures/embedding_projections/` |
| `fig:delta_dist` | `delta_distribution_all.pdf` | `assets/figures/delta_distribution_all.*` |
| `fig:en_split` | `en_in_index_split.pdf` | `assets/figures/en_in_index_split.*` |
| `fig:enzh_triad` | `triad_ENZH.pdf` | `assets/figures/triad_ENZH.*` |
| `fig:hub` | `hub_sweeps.pdf` | `assets/figures/hub_sweeps.*` |
| `fig:mono_alignment` | `mono_alignment_curve.pdf` | `assets/figures/mono_alignment_curve.*` |
| `fig:lambda_star_summary` | `lambda_star_summary.pdf` | `assets/figures/lambda_star_summary.*` |
| `fig:headroom` | `headroom_scatter.pdf` | `assets/figures/headroom_scatter.*` |
| `fig:typology` | `typology_scatter.pdf` | `assets/figures/typology_scatter.*` |

## Table Outputs

| Paper table/topic | Repo output |
| --- | --- |
| Main endpoint-vs-interior delta summaries | `artifacts/tables/full_mmarco_processed_results.csv` |
| Metrics delta table | `query_embedding_mix/calculate_paper_values.py` output |
| Language factor probes | `configs/language_pairs_typology_metrics.csv`, `calculate_paper_values.py` |
| Word-mix validation summary | `artifacts/tables/word_mix_processed.csv` |
| Model-family and Qwen scale ablations | `artifacts/tables/ablation_processed_results.csv` |
