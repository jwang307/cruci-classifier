## Purpose

Train the frozen ESM2-35M crucivirus classifier grid on one CUDA GPU, persist all analysis outputs to Hugging Face, and make the instance safe to terminate after upload.

The grid is:

- domains: `fullCP`, `Rdomain`, `Sdomain`, `Pdomain`
- folds: `Blue`, `Green`, `Purple`, `Red`, `Yellow`
- total models: `20`

## Cloud setup with uv

Use this for an L40S host with driver/CUDA similar to:

```text
Driver Version: 560.35.03
CUDA Version: 12.6
```

Install dependencies:

```bash
cd /workspace/cruci-classifier
git checkout codex/esm-classifier-hardening
git pull

export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
bash scripts/setup_cloud_uv.sh
source .venv/bin/activate
```

Verify CUDA:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY
```

Expected:

```text
torch cuda: 12.6
cuda available: True
gpu: NVIDIA L40S
```

## Data setup

The expected dataset path is:

```text
/workspace/cruci-classifier/data/MachineLearning
```

If starting from the zip:

```bash
cd /workspace/cruci-classifier
mkdir -p data
python -m zipfile -e data/MachineLearning.zip data
```

Audit before launch:

```bash
python scripts/audit_machinelearning_dataset.py \
  --data_root data/MachineLearning \
  --out_dir results/dataset_audit
```

Known audit caveat: the R-domain has exact train/test sequence overlaps with opposite labels for the `Cruci_CruV_493` / `Ribo_ND_132327` pair across Blue/Green. Treat R-domain biological conclusions cautiously.

## Hugging Face setup

Create a write token and log in:

```bash
export HF_TOKEN=hf_...
hf auth login --token "${HF_TOKEN}"
```

Set the target private dataset repo:

```bash
export HF_REPO_ID="<user-or-org>/cruci-esm35m-grid"
export HF_REPO_TYPE=dataset
export HF_PRIVATE=1
```

The upload script creates the repo automatically if the token can create repos in that namespace.

Smoke-test write access:

```bash
python scripts/upload_to_hf.py \
  --repo_id "${HF_REPO_ID}" \
  --repo_type dataset \
  --private \
  --path README.md \
  --path_in_repo smoke_README.md
```

## W&B setup

For live charts:

```bash
wandb login
export WANDB_MODE=online
export WANDB_PROJECT=cruci
```

If `WANDB_MODE=offline`, training still runs and HF artifacts are still uploaded, but W&B charts will not sync live.

## Recommended L40S full run

Use a `tmux` session so the job survives SSH disconnects:

```bash
tmux new -s cruci-grid
```

Inside `tmux`:

```bash
cd /workspace/cruci-classifier
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

mkdir -p "${RESULTS_DIR}"
bash scripts/run_grid.sh 2>&1 | tee "${RESULTS_DIR}/launcher.log"
```

If full-capsid batches OOM on L40S:

```bash
export BATCH_SIZE=16
bash scripts/run_grid.sh 2>&1 | tee -a "${RESULTS_DIR}/launcher.log"
```

Completed folds are skipped unless `FORCE=1`.

## Training details

Model:

- base encoder: `esm2_t12_35M_UR50D`
- representation layer: `12`
- pooling: BOS token embedding, `representations[12][:, 0, :]`
- head: `nn.Linear(hidden_dim, 1)`
- trainable weights: classifier head only
- frozen weights: all ESM parameters, `requires_grad=False`

Data:

- labels: `1 = crucivirus`, `0 = RNA virus`
- sequence cleaning: stop `*` removed by default; unsupported amino acids replaced with `X`
- validation: stratified split from each fold's training CSV
- validation fraction: `0.15`
- seed: `42`
- test set: held-out fold test FASTA, never used for threshold selection

Loss and optimizer:

- loss: `torch.nn.BCEWithLogitsLoss`
- class imbalance: `POS_WEIGHT=auto`, where `pos_weight = n_negative / n_positive`
- optimizer: `torch.optim.Adam`
- optimized parameters: `model.classifier.parameters()`
- learning rate: `1e-3`

Batching:

- recommended L40S batch size: `32`
- fallback batch size: `16`
- dataloader workers: `0`
- train loader shuffled each epoch
- validation/test loaders not shuffled

Stopping:

- no early stopping
- the model trains through all configured `EPOCHS`
- do not set `MAX_STEPS` for real runs
- `MAX_STEPS` is only for smoke tests

Checkpoint selection:

- every epoch logs validation metrics
- `best_head.pt` is selected by maximum validation F1 at threshold `0.5`
- `final_head.pt` is the final epoch checkpoint
- selected reporting threshold is chosen after training by maximizing F1 on validation probabilities
- test metrics are reported at both threshold `0.5` and the selected validation threshold

Step counts:

\[
\text{steps per epoch} = \left\lceil\frac{0.85 \cdot N_\text{train fold}}{\text{batch size}}\right\rceil
\]

For this dataset, folds have roughly `1.9k-2.1k` train examples before validation. With `BATCH_SIZE=32`, expect roughly:

- `50-56` optimizer steps per epoch per model
- `2500-2800` optimizer steps per model for `EPOCHS=50`
- `50k-56k` optimizer steps across the 20-model grid

Exact counts are saved in:

```text
<RESULTS_DIR>/<domain>/<fold>/checkpoints/training_summary.json
```

## Output layout

Per model:

```text
<RESULTS_DIR>/<domain>/<fold>/
  train.log
  checkpoints/
    best_head.pt
    final_head.pt
    training_summary.json
    train_val_split.json
  eval/
    test_metrics.json
    test_eval_results.csv
    test_precision_recall_curve.csv
    test_precision_recall_curve.png
    test_confusion_matrix.png
    test_score_histogram.png
  prepared/
    prepared_summary.json
```

Aggregate files:

```text
<RESULTS_DIR>/aggregate_metrics.csv
<RESULTS_DIR>/all_oof_predictions.csv
<RESULTS_DIR>/run_manifest.json
```

Hugging Face path:

```text
${HF_REPO_ID}/${HF_PATH_PREFIX}/...
```

By default, uploaded artifacts include `best_head.pt` and `final_head.pt`, but exclude bulky full ESM checkpoints and prepared CSVs.

## Monitoring

In the tmux session:

```bash
tail -f "${RESULTS_DIR}/launcher.log"
```

From another shell:

```bash
find "${RESULTS_DIR}" -path '*/eval/test_metrics.json' | wc -l
nvidia-smi
```

Expected completed model count:

```text
20
```

## Resume and rerun

Resume after interruption:

```bash
bash scripts/run_grid.sh 2>&1 | tee -a "${RESULTS_DIR}/launcher.log"
```

The runner skips a fold if this file exists:

```text
<RESULTS_DIR>/<domain>/<fold>/eval/test_metrics.json
```

Force rerun:

```bash
FORCE=1 bash scripts/run_grid.sh 2>&1 | tee -a "${RESULTS_DIR}/launcher.log"
```

## Final upload before terminating instance

Run this after the grid finishes:

```bash
python scripts/upload_to_hf.py \
  --repo_id "${HF_REPO_ID}" \
  --repo_type dataset \
  --private \
  --path "${RESULTS_DIR}" \
  --path_in_repo "${HF_PATH_PREFIX}"
```

Then verify the local count:

```bash
find "${RESULTS_DIR}" -path '*/eval/test_metrics.json' | wc -l
ls -lh "${RESULTS_DIR}/aggregate_metrics.csv" \
       "${RESULTS_DIR}/all_oof_predictions.csv" \
       "${RESULTS_DIR}/run_manifest.json"
```

After HF upload succeeds and the count is `20`, the cloud instance can be terminated.
