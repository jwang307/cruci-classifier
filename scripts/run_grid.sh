#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data/MachineLearning}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/grid_esm35m}"
DOMAINS="${DOMAINS:-fullCP Rdomain Sdomain Pdomain}"
FOLDS="${FOLDS:-Blue Green Purple Red Yellow}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-3}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
POS_WEIGHT="${POS_WEIGHT:-auto}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-cruci}"
SAVE_FULL_CHECKPOINTS="${SAVE_FULL_CHECKPOINTS:-0}"
FORCE="${FORCE:-0}"
MAX_STEPS="${MAX_STEPS:-}"
LIMIT_TRAIN_EXAMPLES="${LIMIT_TRAIN_EXAMPLES:-}"
LIMIT_TEST_EXAMPLES="${LIMIT_TEST_EXAMPLES:-}"

HF_REPO_ID="${HF_REPO_ID:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_PRIVATE="${HF_PRIVATE:-0}"
HF_PATH_PREFIX="${HF_PATH_PREFIX:-grid_esm35m}"
HF_UPLOAD_EACH="${HF_UPLOAD_EACH:-0}"
HF_UPLOAD_FINAL="${HF_UPLOAD_FINAL:-0}"

mkdir -p "${RESULTS_DIR}"

extra_train_args=()
if [[ "${SAVE_FULL_CHECKPOINTS}" != "1" ]]; then
  extra_train_args+=(--no_save_full_checkpoint)
fi
if [[ -n "${MAX_STEPS}" ]]; then
  extra_train_args+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${LIMIT_TRAIN_EXAMPLES}" ]]; then
  extra_train_args+=(--limit_train_examples "${LIMIT_TRAIN_EXAMPLES}")
fi
if [[ -n "${LIMIT_TEST_EXAMPLES}" ]]; then
  extra_train_args+=(--limit_test_examples "${LIMIT_TEST_EXAMPLES}")
fi

upload_args=()
if [[ "${HF_PRIVATE}" == "1" ]]; then
  upload_args+=(--private)
fi

run_hf_upload() {
  local local_path="$1"
  local remote_path="$2"
  if [[ -z "${HF_REPO_ID}" ]]; then
    echo "HF_REPO_ID unset; skipping upload for ${local_path}"
    return 0
  fi
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/upload_to_hf.py" \
    --repo_id "${HF_REPO_ID}" \
    --repo_type "${HF_REPO_TYPE}" \
    --path "${local_path}" \
    --path_in_repo "${remote_path}" \
    "${upload_args[@]}"
}

echo "root=${ROOT_DIR}"
echo "data_root=${DATA_ROOT}"
echo "results_dir=${RESULTS_DIR}"
echo "domains=${DOMAINS}"
echo "folds=${FOLDS}"
echo "device=${DEVICE} batch_size=${BATCH_SIZE} epochs=${EPOCHS}"

for domain in ${DOMAINS}; do
  for fold in ${FOLDS}; do
    run_dir="${RESULTS_DIR}/${domain}/${fold}"
    prepared_dir="${run_dir}/prepared"
    checkpoint_dir="${run_dir}/checkpoints"
    eval_dir="${run_dir}/eval"
    metrics_json="${eval_dir}/test_metrics.json"
    log_file="${run_dir}/train.log"

    if [[ -s "${metrics_json}" && "${FORCE}" != "1" ]]; then
      echo "Skipping ${domain}/${fold}; found ${metrics_json}. Set FORCE=1 to rerun."
      continue
    fi

    mkdir -p "${run_dir}" "${checkpoint_dir}" "${eval_dir}"
    echo "Preparing ${domain}/${fold}"
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_fold_csvs.py" \
      --data_root "${DATA_ROOT}" \
      --out_dir "${prepared_dir}" \
      --domain "${domain}" \
      --fold "${fold}"

    train_csv="${prepared_dir}/${domain}/${fold}/train.csv"
    test_csv="${prepared_dir}/${domain}/${fold}/test.csv"
    wandb_name="esm35m_${domain}_${fold}"

    echo "Training ${domain}/${fold}"
    "${PYTHON_BIN}" "${ROOT_DIR}/classifier/train.py" \
      --train_csv "${train_csv}" \
      --test_csv "${test_csv}" \
      --checkpoint_dir "${checkpoint_dir}" \
      --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --val_fraction "${VAL_FRACTION}" \
      --pos_weight "${POS_WEIGHT}" \
      --device "${DEVICE}" \
      --wandb_mode "${WANDB_MODE}" \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_name "${wandb_name}" \
      --save_every 0 \
      "${extra_train_args[@]}" 2>&1 | tee "${log_file}"

    selected_threshold="$("${PYTHON_BIN}" - "${checkpoint_dir}/training_summary.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print(payload["selected_val_threshold"])
PY
)"

    echo "Evaluating ${domain}/${fold} with selected_threshold=${selected_threshold}"
    "${PYTHON_BIN}" "${ROOT_DIR}/classifier/eval.py" \
      --csv "${test_csv}" \
      --checkpoint "${checkpoint_dir}/best_head.pt" \
      --batch_size "${BATCH_SIZE}" \
      --out_dir "${eval_dir}" \
      --device "${DEVICE}" \
      --selected_threshold "${selected_threshold}"

    if [[ "${HF_UPLOAD_EACH}" == "1" ]]; then
      run_hf_upload "${run_dir}" "${HF_PATH_PREFIX}/${domain}/${fold}"
    fi
  done
done

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/aggregate_grid_results.py" --root "${RESULTS_DIR}"

if [[ "${HF_UPLOAD_FINAL}" == "1" ]]; then
  run_hf_upload "${RESULTS_DIR}" "${HF_PATH_PREFIX}"
fi
