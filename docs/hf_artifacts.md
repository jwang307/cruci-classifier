## Goal

Persist cloud-run outputs outside the rented GPU instance.

The default path is a Hugging Face Hub **dataset** repo containing:

- tiny head-only checkpoints: `checkpoints/best_head.pt`
- training metadata: `checkpoints/training_summary.json`, `train_val_split.json`
- evaluation tables: `eval/test_eval_results.csv`
- metrics and plots: `eval/test_metrics.json`, PR curve, confusion matrix, histogram
- aggregate analysis files: `aggregate_metrics.csv`, `all_oof_predictions.csv`
- run manifest: `run_manifest.json`

Legacy full ESM checkpoints are skipped by default because they duplicate the public ESM weights.

## One-time Hugging Face setup

Create a token with write permission at:

```text
https://huggingface.co/settings/tokens
```

On the cloud instance:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install fair-esm pandas scikit-learn matplotlib biopython wandb huggingface_hub

export HF_TOKEN=hf_...
hf auth login --token "${HF_TOKEN}"
```

Use a private HF repo unless the sequences/results are intended to be public.

## Canary run with upload

```bash
export PYTHON_BIN="$(pwd)/.venv/bin/python"
export DATA_ROOT="$(pwd)/data/MachineLearning"
export RESULTS_DIR="$(pwd)/results/grid_esm35m"
export DEVICE=cuda
export BATCH_SIZE=16
export EPOCHS=2
export DOMAINS=fullCP
export FOLDS=Blue
export WANDB_MODE=offline
export MAX_STEPS=20

export HF_REPO_ID="<user-or-org>/cruci-esm35m-grid"
export HF_REPO_TYPE=dataset
export HF_PRIVATE=1
export HF_UPLOAD_EACH=1
export HF_UPLOAD_FINAL=1

bash scripts/run_grid.sh
```

Unset `MAX_STEPS` for a real canary that trains full epochs:

```bash
unset MAX_STEPS
```

## Full sequential grid

```bash
export PYTHON_BIN="$(pwd)/.venv/bin/python"
export DATA_ROOT="$(pwd)/data/MachineLearning"
export RESULTS_DIR="$(pwd)/results/grid_esm35m"
export DEVICE=cuda
export BATCH_SIZE=32
export EPOCHS=5
export WANDB_MODE=offline

export HF_REPO_ID="<user-or-org>/cruci-esm35m-grid"
export HF_REPO_TYPE=dataset
export HF_PRIVATE=1
export HF_UPLOAD_EACH=1
export HF_UPLOAD_FINAL=1

tmux new -s cruci-grid
bash scripts/run_grid.sh
```

Use `BATCH_SIZE=16` on 24GB GPUs if full capsid runs OOM.

## Resume behavior

`scripts/run_grid.sh` skips a `domain/fold` if this exists:

```text
results/grid_esm35m/<domain>/<fold>/eval/test_metrics.json
```

Rerun from scratch with:

```bash
FORCE=1 bash scripts/run_grid.sh
```

## Upload-only command

Upload an existing local result directory:

```bash
python scripts/upload_to_hf.py \
  --repo_id "<user-or-org>/cruci-esm35m-grid" \
  --repo_type dataset \
  --private \
  --path results/grid_esm35m \
  --path_in_repo grid_esm35m
```

By default this ignores:

- `best.pt`
- `epoch_*.pt`
- prepared CSVs
- Python caches

To include full ESM checkpoints:

```bash
python scripts/upload_to_hf.py \
  --repo_id "<user-or-org>/cruci-esm35m-grid" \
  --repo_type dataset \
  --path results/grid_esm35m \
  --path_in_repo grid_esm35m \
  --include_full_checkpoints
```

## Download results later

```bash
hf download "<user-or-org>/cruci-esm35m-grid" \
  --repo-type dataset \
  --local-dir results_from_hf
```

The analysis-critical files are:

```text
grid_esm35m/aggregate_metrics.csv
grid_esm35m/all_oof_predictions.csv
grid_esm35m/run_manifest.json
grid_esm35m/*/*/eval/test_metrics.json
grid_esm35m/*/*/eval/test_eval_results.csv
grid_esm35m/*/*/checkpoints/best_head.pt
```
