#!/usr/bin/env bash
# Quick regression check for residual connection patterns using train_classifier.py.
#
# Usage:
#   1. Activate the desired Python environment.
#   2. bash scripts/run_node_debug_checks.sh
#
# The script launches short runs of train_classifier.py with different
# residual pattern configurations and confirms that the newly introduced
# pattern parameters (alpha/theta as scalars or conv1x1 projections) are
# materialized in the experiment log.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)
cd "${PROJECT_ROOT}"

TORCHRUN_BIN=${TORCHRUN_BIN:-$(command -v torchrun)}
if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not found in PATH." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_ROOT}/results-classifier-debug}
MAX_LOG_LINES=${MAX_LOG_LINES:-5}
# export WANDB_MODE=offline
# export WANDB_DISABLED=1

run_train() {
  local nproc=$1
  local recipe=$2
  local method=$3
  local pattern=$4
  local steps=$5
  local rescale_mode=${6:-}

  local tag="${method}-${pattern}"
  if [[ "${pattern}" == "rescale_stream" && -n "${rescale_mode}" ]]; then
    tag+="-${rescale_mode}"
  fi

  local case_root="${RESULTS_ROOT}/${tag}"
  mkdir -p "${case_root}"

  local cmd=(
    "${TORCHRUN_BIN}"
    --standalone
    "--nproc_per_node=${nproc}"
    train_classifier.py
    --config_file "${recipe}"
    --results_dir "${case_root}"
    --epochs 1
    --max_train_steps "${steps}"
    --log_interval 1
    --save_every_steps $((steps * 1000))
    --save_every_epochs 1000
    # --debug
  )

  if [[ "${method}" == "orthogonal" ]]; then
    cmd+=(--orthogonal_residual)
  else
    cmd+=(--no_orthogonal_residual)
  fi

  if [[ "${pattern}" != "default" ]]; then
    cmd+=(--residual_pattern "${pattern}")
  fi

  if [[ "${pattern}" == "rescale_stream" && -n "${rescale_mode}" ]]; then
    cmd+=(--residual_rescale_mode "${rescale_mode}")
  fi

  echo "\n==== Running: nproc=${nproc} recipe=${recipe} method=${method} pattern=${pattern} rescale_mode=${rescale_mode:-n/a} ===="
  echo "Command: ${cmd[*]}"
  "${cmd[@]}"
}

check_log_contains() {
  local log_file=$1
  shift
  local missing=0
  for needle in "$@"; do
    if ! grep -q "${needle}" "${log_file}"; then
      echo "[ERROR] Expected string '${needle}' not found in ${log_file}" >&2
      missing=1
    fi
  done
  return ${missing}
}

run_and_validate() {
  local nproc=$1
  local recipe=$2
  local method=$3
  local pattern=$4
  local steps=$5
  local rescale_mode=${6:-}
  shift $(( $# >= 6 ? 6 : 5 ))
  local expectations=("$@")

  run_train "${nproc}" "${recipe}" "${method}" "${pattern}" "${steps}" "${rescale_mode}"

  local tag="${method}-${pattern}"
  if [[ "${pattern}" == "rescale_stream" && -n "${rescale_mode}" ]]; then
    tag+="-${rescale_mode}"
  fi
  local latest_run
  latest_run=$(ls -td "${RESULTS_ROOT}/${tag}"/* 2>/dev/null | head -n1 || true)
  if [[ -z "${latest_run}" ]]; then
    echo "[ERROR] No results directory created for tag ${tag}." >&2
    return 1
  fi

  local log_file="${latest_run}/log.txt"
  if [[ ! -f "${log_file}" ]]; then
    echo "[ERROR] Missing log.txt in ${latest_run}" >&2
    return 1
  fi

  echo "-- Tail of ${log_file} --"
  tail -n "${MAX_LOG_LINES}" "${log_file}"

  if ! check_log_contains "${log_file}" "${expectations[@]}"; then
    return 1
  fi

  echo "[OK] Expected pattern parameters logged for ${tag}."
}

run_and_validate 1 recipes/debug_vit.yaml linear rezero 6 '' "attn_alpha" "mlp_alpha"
run_and_validate 1 recipes/debug_resnet.yaml orthogonal rezero_constrained 6 '' "conv_theta"
run_and_validate 1 recipes/debug_resnet.yaml orthogonal rescale_stream 6 scalar "conv_rescale_alpha"
run_and_validate 1 recipes/debug_resnet.yaml orthogonal rescale_stream 6 conv1x1 "conv_rescale_proj"

echo "\nAll checks completed. Logs stored under ${RESULTS_ROOT}."
