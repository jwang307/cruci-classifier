## Goal

Derisk the next crucivirus-vs-RNA-virus ESM2-35M classifier experiments on the new `MachineLearning` five-cluster dataset before launching longer GPU runs.

The model path remains intentional: **frozen ESM2-35M encoder plus a lightweight trained binary classification head**.

## Data audit findings

Input dataset: `data/MachineLearning`.

| Check | Result |
|---|---:|
| FASTA records checked | 49,420 |
| Domain/fold/split count rows | 40 |
| Stop characters cleaned | 10,140 |
| Unsupported residues cleaned | 0 |
| Records requiring cleanup | 10,120 |
| Train/test ID or exact-sequence overlaps | 2 |
| Within-file duplicate ID/exact-sequence groups | 1,193 |

Main red flags:

- Full-capsid FASTAs contain many `*` stop characters. The pipeline now removes these before ESM tokenization.
- Red and Purple test folds are highly imbalanced. Accuracy alone is not meaningful.
- Two exact R-domain sequence overlaps cross train/test with opposite labels:
  - Blue R-domain: train `Cruci_CruV_493`, test `Ribo_ND_132327`
  - Green R-domain: train `Ribo_ND_132327`, test `Cruci_CruV_493`
- Duplicate exact sequences are common within files, especially among related Cruci IDs. This should be reported and considered when interpreting generalization.

Audit files:

- `results/dataset_audit/dataset_summary.csv`
- `results/dataset_audit/sequence_counts_by_fold_domain_class.csv`
- `results/dataset_audit/length_summary_by_fold_domain_class.csv`
- `results/dataset_audit/duplicate_sequences.tsv`
- `results/dataset_audit/train_test_overlaps.tsv`
- `results/dataset_audit/nonstandard_residues.tsv`

## Local MPS smoke run

Input: `fullCP` / `Blue` fold.

Training command used:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run \
  --with torch --with fair-esm --with pandas --with scikit-learn --with wandb \
  python classifier/train.py \
  --train_csv results/local_smoke_esm35m/prepared/fullCP/Blue/train.csv \
  --test_csv results/local_smoke_esm35m/prepared/fullCP/Blue/test.csv \
  --checkpoint_dir /tmp/cruci_local_smoke_esm35m_fullcp_blue_ckpt \
  --batch_size 8 \
  --epochs 3 \
  --lr 1e-3 \
  --val_fraction 0.15 \
  --pos_weight auto \
  --device mps \
  --wandb_mode disabled \
  --save_every 0
```

Split and training details:

| Field | Value |
|---|---:|
| Train examples after internal split | 1,764 |
| Validation examples | 312 |
| Validation positives / negatives | 115 / 197 |
| Test examples | 407 |
| Test positives / negatives | 238 / 169 |
| Training steps | 663 |
| Selected validation threshold | 0.478096 |

Validation metrics by epoch:

| Epoch | Steps | Loss | Val F1 | Val balanced accuracy | Val AUROC |
|---:|---:|---:|---:|---:|---:|
| 1 | 221 | 0.8242 | 0.6173 | 0.6581 | 0.8535 |
| 2 | 442 | 0.7332 | 0.7586 | 0.8069 | 0.8702 |
| 3 | 663 | 0.6767 | 0.7328 | 0.7879 | 0.8836 |

## Local test results

Blue full-capsid held-out fold at default threshold 0.5:

| Metric | Value |
|---|---:|
| Accuracy | 0.7985 |
| Balanced accuracy | 0.8080 |
| MCC | 0.6071 |
| Precision | 0.8861 |
| Recall | 0.7521 |
| Specificity | 0.8639 |
| F1 | 0.8136 |
| AUROC | 0.8587 |
| AUPRC / average precision | 0.9042 |

At validation-selected threshold 0.478096:

| Metric | Value |
|---|---:|
| Accuracy | 0.7887 |
| Balanced accuracy | 0.7867 |
| MCC | 0.5693 |
| Precision | 0.8333 |
| Recall | 0.7983 |
| Specificity | 0.7751 |
| F1 | 0.8155 |
| AUROC | 0.8587 |
| AUPRC / average precision | 0.9042 |

Interpretation:

- The local MacBook MPS run is functional for several hundred frozen-ESM head-training steps.
- The model learns real signal on the Blue full-capsid fold within 3 epochs.
- The validation split is large enough for smoke-run conclusions: 312 examples with both classes represented.
- AUROC and AUPRC are strong enough to justify the larger H200 experiment.
- Threshold choice shifts precision/recall, but continuous score quality is unchanged.

## Code readiness updates

Implemented derisking changes:

- ESM-safe protein cleanup for FASTA and CSV ingestion.
- Stratified internal validation split with class-count logging.
- Optional explicit `--val_csv` for fully controlled validation files.
- MPS/CUDA/CPU device selection.
- W&B `online`/`offline`/`disabled` mode.
- Robust metrics for class imbalance and single-class edge cases.
- Validation-selected threshold reporting without using test labels.
- Evaluation outputs include prediction columns at 0.5 and optional validation-selected threshold.
- Fold CSV preparation script for the MachineLearning dataset.
- Dataset audit script and audit artifacts.

## Recommended H200 run plan

Start with the same Blue full-capsid fold to verify the GPU environment, then run all five full-capsid folds, then R/S/P domains.

Environment smoke command for one H200:

```bash
uv run --with torch --with fair-esm --with pandas --with scikit-learn --with wandb \
  python scripts/prepare_fold_csvs.py \
  --data_root data/MachineLearning \
  --out_dir results/prepared_folds \
  --domain fullCP \
  --fold Blue

WANDB_MODE=offline uv run --with torch --with fair-esm --with pandas --with scikit-learn --with wandb \
  python classifier/train.py \
  --train_csv results/prepared_folds/fullCP/Blue/train.csv \
  --test_csv results/prepared_folds/fullCP/Blue/test.csv \
  --checkpoint_dir results/models/checkpoints/fullCP/Blue \
  --batch_size 16 \
  --epochs 10 \
  --lr 1e-3 \
  --val_fraction 0.15 \
  --pos_weight auto \
  --device cuda \
  --wandb_mode offline
```

Full run grid:

```text
input domains: fullCP, Rdomain, Sdomain, Pdomain
folds: Blue, Green, Purple, Red, Yellow
model: frozen ESM2-35M + binary head
metrics: accuracy, balanced accuracy, MCC, precision, recall, specificity, F1, AUROC, AUPRC
```

Suggested H200 defaults:

| Parameter | Value |
|---|---:|
| GPU | 1x H200 |
| Batch size | 16, increase to 32 if memory allows |
| Epochs | 10 initially |
| LR | 1e-3 |
| Validation fraction | 0.15 stratified inside train clusters |
| Class weighting | `--pos_weight auto` |
| Checkpoint selection | best validation F1 at default threshold |
| Reporting | use continuous scores; thresholds are secondary |

## Caveats before biological interpretation

- The R-domain exact-sequence cross-label overlap should be reviewed with Nacho before treating R-domain results as final.
- The severe Red/Purple fold imbalance means fold-level MCC, balanced accuracy, AUROC, and AUPRC matter more than raw accuracy.
- Duplicate exact sequences within train/test files are common; if results are surprisingly high, run sequence-identity clustered controls next.
- Baselines from the plan are still needed: length-only, amino-acid composition, k-mer, frozen-embedding logistic regression, and label-shuffle controls.
