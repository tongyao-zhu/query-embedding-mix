# Project Structure

The repo separates source code, experiment entrypoints, compact artifacts, and generated runtime outputs.

## Source Code

```text
query_embedding_mix/
```

Python scripts live here. Most are runnable CLI entrypoints because the project is optimized for paper reproduction rather than packaging.

## Experiment Entrypoints

```text
scripts/
```

Shell launchers define the main experiment grids, GPU scheduling choices, default paths, and collection workflows.

## Configs

```text
configs/
|-- language_pairs_typology_metrics.csv
`-- qid_lists/
```

Small metadata and qid filters that are required to interpret or reproduce the paper are tracked.

## Compact Artifacts

```text
artifacts/
|-- tables/
|-- analysis/
`-- examples/
```

Tracked artifacts should be small enough to review in git. See [ARTIFACTS.md](ARTIFACTS.md).

## Figures

```text
assets/figures/
```

Final paper figures and README-visible figures live here.

## Generated Runtime Outputs

The following directories are ignored and should be regenerated locally:

```text
data/
indexes/
runs/
results/
logs/
index_logs/
index_logs_ablation/
```

## Naming Conventions

- Use ISO language codes in generated files when possible: `en`, `zh`, `vi`, `hi`, `id`.
- Use uppercase pair labels in tables: `EN-ZH`.
- Use lower-case pair labels in paths: `en-zh`.
- Keep model and dataset tags in generated directory names so multiple runs can coexist.

## Where To Put New Work

| Change | Location |
| --- | --- |
| New experiment Python code | `query_embedding_mix/` |
| New full-run launcher | `scripts/` |
| New compact result table | `artifacts/tables/` |
| New final figure | `assets/figures/` |
| New qid filter or small metadata file | `configs/` |
| New documentation | `docs/` |
| Safe local configuration template | `.env.example` |
| Full raw output | Ignored runtime directories |
