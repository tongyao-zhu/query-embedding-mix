# Artifacts

This repository is designed to be useful without committing large runtime products. It includes compact, paper-facing artifacts and ignores full raw runs.

## Checked In

| Path | Purpose |
| --- | --- |
| `artifacts/tables/full_mmarco_results.csv` | Collected main BGE-M3 results before final processing. |
| `artifacts/tables/full_mmarco_processed_results.csv` | Main table with endpoint best, best interior mixture, delta, lambda star, and metadata. The checked-in table includes the expanded 91-pair/273-setting grid. |
| `artifacts/tables/ablation_results.csv` | Collected ablation results. |
| `artifacts/tables/ablation_processed_results.csv` | Processed model-family and scale ablation table. |
| `artifacts/tables/word_mix_curves.csv` | Word-mix and embedding-mix curves for validation pairs. |
| `artifacts/tables/word_mix_processed.csv` | Processed word-mix validation summary. |
| `artifacts/tables/paper_values.txt` | Printed values used while writing the paper. |
| `artifacts/analysis/en_zh_embedding_space/` | EN-ZH geometry diagnostics and visualizations. |
| `artifacts/examples/repro_en_zh_example/` | Small EN-ZH example result bundle for readers. |
| `assets/figures/` | Paper figures generated from compact tables and analyses. |

## Not Checked In

These are expected to be generated locally:

| Path | Contents |
| --- | --- |
| `data/` | Downloaded mMARCO queries, generated code-mixed queries, qrels caches. |
| `indexes/` | FAISS indexes and document-id maps. |
| `runs/` | TREC run files and intermediate run products. |
| `results/` | Raw evaluation outputs before collection. |
| `logs/` | Scheduler, encoding, retrieval, and API-generation logs. |
| `index_logs/` | Full-index encoding logs. |
| `index_logs_ablation/` | Ablation-index encoding logs. |

## Regenerating Tables

Main full mMARCO table:

```bash
python query_embedding_mix/collect_results.py \
  ./results/mmarco_full \
  --output ./artifacts/tables/full_mmarco_results.csv \
  --processed-out ./artifacts/tables/full_mmarco_processed_results.csv
```

Ablation table:

```bash
python query_embedding_mix/collect_ablation_results.py \
  ./results/mmarco_full/ablation2 \
  --output ./artifacts/tables/ablation_results.csv \
  --processed-out ./artifacts/tables/ablation_processed_results.csv
```

Paper values:

```bash
python query_embedding_mix/calculate_paper_values.py
```

Figures:

```bash
python query_embedding_mix/plot_paper_figures.py
```

## Adding New Artifacts

Please keep artifacts small and reviewable:

- Prefer compact CSV, JSON, Markdown, text, PNG, or PDF files.
- Include the command or script that generated the artifact.
- Avoid committing raw benchmark data, private API generations, full FAISS indexes, or logs.
- Put final paper figures under `assets/figures/`.
- Put compact numeric outputs under `artifacts/tables/`.
