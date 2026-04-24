# Section 5: Ablation

The ablation section tests whether the main mixing patterns are specific to BGE-M3 or hold across model families and scales.

## Commands

```bash
bash scripts/run_encode_index_ablation.sh
bash scripts/run_ablation.sh
```

Collect ablation tables:

```bash
python query_embedding_mix/collect_ablation_results.py \
  ./results/mmarco_full/ablation2 \
  --output ./artifacts/tables/ablation_results.csv \
  --processed-out ./artifacts/tables/ablation_processed_results.csv
```

## Relevant Files

- `scripts/run_encode_index_ablation.sh`
- `scripts/run_ablation.sh`
- `query_embedding_mix/collect_ablation_results.py`
- `artifacts/tables/ablation_results.csv`
- `artifacts/tables/ablation_processed_results.csv`
- `assets/figures/ablation_hub_DE.png`
- `assets/figures/ablation_hub_ZH.png`
- `assets/figures/qwen_scale.png`
