# Ablations

This folder collects the model-family and scale checks. They are useful if you want to know whether the main patterns are specific to BGE-M3, but they are separate from the core README so the first page can stay focused on the main result.

## What The Ablations Test

- Model family: whether the English-boundary and strongest-partner patterns hold beyond BGE-M3.
- Model scale: whether Qwen3 embedding models show similar behavior across sizes.
- Smaller retrieval setting: ablations run on 100k-passage subsets to keep compute tractable.

## Commands

```bash
bash scripts/run_encode_index_ablation.sh
bash scripts/run_ablation.sh
```

Collect processed tables:

```bash
python query_embedding_mix/collect_ablation_results.py \
  ./results/mmarco_full/ablation2 \
  --output ./artifacts/tables/ablation_results.csv \
  --processed-out ./artifacts/tables/ablation_processed_results.csv
```

## Checked-In Outputs

| Artifact | Contents |
| --- | --- |
| `artifacts/tables/ablation_results.csv` | Raw collected ablation metrics. |
| `artifacts/tables/ablation_processed_results.csv` | Processed endpoint-vs-interior summaries. |
| `assets/figures/ablation_hub_DE.png` | Model-family comparison for DE document settings. |
| `assets/figures/ablation_hub_ZH.png` | Model-family comparison for ZH document settings. |
| `assets/figures/qwen_scale.png` | Qwen3 scale comparison. |

## Related Paper Map

For label-level correspondence with the LaTeX source, see [../paper/05_ABLATION.md](../paper/05_ABLATION.md) and [../paper/LATEX_SOURCE_MAP.md](../paper/LATEX_SOURCE_MAP.md).
