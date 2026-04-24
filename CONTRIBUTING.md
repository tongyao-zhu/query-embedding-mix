# Contributing

Thank you for helping make this research repo easier to reproduce and extend.

## Good First Contributions

- Clarify setup or reproduction instructions.
- Add missing command examples for a specific experiment path.
- Improve result collection, validation, or plotting robustness.
- Add small smoke tests for parsers, collectors, or plotting utilities.
- Report mismatches between the paper, checked-in artifacts, and scripts.

## Development Setup

```bash
conda create -n query-mix python=3.11.13 -y
conda activate query-mix
conda install -c conda-forge faiss-gpu=1.8.0 -y
pip install -r requirements.txt
```

For documentation-only changes, no GPU setup is needed.

## Pull Request Checklist

- Keep generated data out of git unless it is a compact paper-facing artifact.
- Do not commit raw mMARCO data, FAISS indexes, full run folders, logs, caches, or API credentials.
- Prefer environment variables over hard-coded local paths.
- Include the command you used to validate the change.
- Keep edits focused on one purpose per pull request.

## Code Style

This repo is primarily research code. Please keep the existing direct script style unless a refactor clearly improves reproducibility.

- Use explicit CLI arguments for new experiment settings.
- Keep output paths configurable.
- Add short comments only when they explain non-obvious experiment logic.
- Preserve current result schemas unless a migration is documented.

## Data and Model Policy

The repository should stay lightweight. Large or redistributable-restricted assets belong outside git:

- `data/`
- `indexes/`
- `runs/`
- `results/`
- `logs/`
- `index_logs/`
- `index_logs_ablation/`

If a new artifact is needed for the paper or documentation, prefer a compact CSV, JSON, text report, or small figure.

## Reporting Issues

When reporting a bug, please include:

- The command you ran.
- The relevant environment variables.
- Python, CUDA, PyTorch, FAISS, and GPU details when applicable.
- A short excerpt of the error log.
- Whether the failure affects full mMARCO, 100k ablations, word-mix validation, or plotting only.
