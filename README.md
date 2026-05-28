## Crucivirus ESM classifier

Frozen ESM2-35M capsid/domain classifier for crucivirus-vs-RNA-virus labels.

This repo currently trains only a lightweight classifier head on top of frozen ESM embeddings.

## Main cloud runbook

Use this for the L40S grid run:

```text
docs/l40s_grid_training.md
```

It covers:

- uv-based CUDA install
- MachineLearning dataset placement
- Hugging Face artifact upload
- W&B online/offline tracking
- full `domain x fold` grid launch
- training hyperparameters and checkpoint semantics
- resume and final upload checks

## Core scripts

```text
scripts/setup_cloud_uv.sh              # create .venv and install CUDA/Python deps with uv
scripts/run_grid.sh                    # sequential 4-domain x 5-fold training/eval/upload runner
scripts/prepare_fold_csvs.py           # clean FASTA fold/domain data into ESM-ready CSVs
scripts/audit_machinelearning_dataset.py
scripts/aggregate_grid_results.py      # aggregate per-fold metrics and OOF predictions
scripts/upload_to_hf.py                # upload durable artifacts to HF Hub
classifier/train.py                    # frozen ESM + linear head training
classifier/eval.py                     # checkpoint evaluation and plots/tables
```

## Quick L40S launch

```bash
cd /workspace/cruci-classifier
git checkout codex/esm-classifier-hardening
git pull

export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
bash scripts/setup_cloud_uv.sh
source .venv/bin/activate

export PYTHON_BIN="$(pwd)/.venv/bin/python"
export DATA_ROOT="$(pwd)/data/MachineLearning"
export RESULTS_DIR="$(pwd)/results/grid_esm35m_e50"
export DEVICE=cuda
export BATCH_SIZE=32
export EPOCHS=50
export LR=1e-3
export VAL_FRACTION=0.15
export POS_WEIGHT=auto
export WANDB_MODE=online
export WANDB_PROJECT=cruci
export SAVE_FULL_CHECKPOINTS=0
unset DOMAINS FOLDS MAX_STEPS LIMIT_TRAIN_EXAMPLES LIMIT_TEST_EXAMPLES

export HF_REPO_ID="<user-or-org>/cruci-esm35m-grid"
export HF_REPO_TYPE=dataset
export HF_PRIVATE=1
export HF_PATH_PREFIX=grid_esm35m_e50
export HF_UPLOAD_EACH=1
export HF_UPLOAD_FINAL=1
export HF_UPLOAD_RETRIES=3

bash scripts/run_grid.sh 2>&1 | tee "${RESULTS_DIR}/launcher.log"
```

## Expected outputs

Per model:

```text
<RESULTS_DIR>/<domain>/<fold>/checkpoints/best_head.pt
<RESULTS_DIR>/<domain>/<fold>/checkpoints/final_head.pt
<RESULTS_DIR>/<domain>/<fold>/checkpoints/training_summary.json
<RESULTS_DIR>/<domain>/<fold>/eval/test_metrics.json
<RESULTS_DIR>/<domain>/<fold>/eval/test_eval_results.csv
```

Aggregates:

```text
<RESULTS_DIR>/aggregate_metrics.csv
<RESULTS_DIR>/all_oof_predictions.csv
<RESULTS_DIR>/run_manifest.json
```
