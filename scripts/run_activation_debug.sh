#!/usr/bin/env bash
# Quick automation for activation metric debug & multi-GPU run.
# Usage examples (run AFTER ssh node03):
#   bash scripts/run_activation_debug.sh --smoke
#   bash scripts/run_activation_debug.sh --multi --config recipes/debug.yaml
#   bash scripts/run_activation_debug.sh --multi --nproc 8 --dataset cifar10 --model vit --preset S \
#        --epochs 1 --global-batch 512 --log-interval 5 --act-log-interval 1 --force
#
# The script will:
#  1. Verify micromamba <env> & torch GPU
#  2. Run a smoke (single GPU) forced activation stats test unless skipped
#  3. Verify activation metrics appear in the log
#  4. Optionally launch a multi-GPU run (torchrun)
#  5. Summarize key metrics (activation means, train loss lines)
#
# Requirements: micromamba env named 'diffusers' containing torch, timm, datasets, wandb.

set -euo pipefail

#############################################
# Defaults
#############################################
DATASET="cifar10"
MODEL="vit"
PRESET="S"
EPOCHS=1
GLOBAL_BATCH=256
NPROC=1
LOG_INTERVAL=2
FORCE_STATS=0
CONFIG_FILE=""
MULTI=0
SMOKE=0
WANDB_MODE_AUTO=1
EXTRA_ARGS=()
RESULTS_DIR="results-classifier"
ENV_NAME="diffusers"
PYTHON_BIN=""
RUN_TORCHRUN=""

COLOR_Y='\033[33m'
COLOR_G='\033[32m'
COLOR_R='\033[31m'
COLOR_C='\033[36m'
COLOR_RESET='\033[0m'

info() { echo -e "${COLOR_G}[INFO]${COLOR_RESET} $*"; }
warn() { echo -e "${COLOR_Y}[WARN]${COLOR_RESET} $*"; }
err()  { echo -e "${COLOR_R}[ERR ]${COLOR_RESET} $*"; }
cmd()  { echo -e "${COLOR_C}$ ${COLOR_RESET}$*"; "$@"; }

#############################################
# Parse args
#############################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --preset) PRESET="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --global-batch) GLOBAL_BATCH="$2"; shift 2;;
    --nproc) NPROC="$2"; shift 2;;
    --log-interval) LOG_INTERVAL="$2"; shift 2;;
    --force) FORCE_STATS=1; shift;;
    --config) CONFIG_FILE="$2"; shift 2;;
    --multi) MULTI=1; shift;;
    --smoke) SMOKE=1; shift;;
    --no-wandb-offline) WANDB_MODE_AUTO=0; shift;;
  --results-dir) RESULTS_DIR="$2"; shift 2;;
  --env) ENV_NAME="$2"; shift 2;;
    --) shift; break;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

#############################################
# Env checks
#############################################
if ! command -v micromamba >/dev/null 2>&1; then
  err "micromamba not found in PATH."; exit 1
fi

if ! micromamba env list | awk '{print $1}' | grep -qw "$ENV_NAME"; then
  err "micromamba env '$ENV_NAME' not found. Available:"; micromamba env list; exit 1
fi

# Non-interactive ssh 세션에서도 확실히 PATH/hook 반영
if ! command -v python >/dev/null 2>&1; then
  # 혹시라도 PATH가 정리 안 됐으면 사용자 bashrc 로드 (타이포 .basrc 예방)
  if [ -f "$HOME/.bashrc" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.bashrc"
  fi
fi

PYTHON_BIN="micromamba run -n ${ENV_NAME} python"
RUN_TORCHRUN="micromamba run -n ${ENV_NAME} torchrun"

# Basic python + torch check
info "Checking python & torch in env: ${ENV_NAME}"
$PYTHON_BIN - <<'PYCHK'
import torch, sys
print('Python', sys.version)
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'gpus=', torch.cuda.device_count())
if not torch.cuda.is_available():
    print('WARNING: CUDA not available. Training will be CPU-bound.')
PYCHK

#############################################
# Decide offline/online wandb
#############################################
if [[ $WANDB_MODE_AUTO -eq 1 && -z "${WANDB_MODE:-}" ]]; then
  export WANDB_MODE=offline
  info "WANDB_MODE=offline (override with --no-wandb-offline to use online)."
fi

STAMP=$(date +%Y%m%d-%H%M%S)

#############################################
# Smoke test (single GPU forced activation)
#############################################
if [[ $SMOKE -eq 1 || $MULTI -eq 0 ]]; then
  info "Running smoke test (single GPU) with forced activation stats"
  FORCE_FLAG=""
  [[ $FORCE_STATS -eq 1 ]] && FORCE_FLAG="--force_activation_stats"
  cmd $RUN_TORCHRUN --nproc-per-node=1 train_classifier.py \
    --dataset "$DATASET" --model "$MODEL" --preset "$PRESET" \
    --epochs $EPOCHS --global_batch_size $GLOBAL_BATCH \
  --log_interval $LOG_INTERVAL \
    --log_activations $FORCE_FLAG --debug "${EXTRA_ARGS[@]}"
fi

LATEST_EXP=$(ls -td ${RESULTS_DIR}/* 2>/dev/null | head -1 || true)
if [[ -z "$LATEST_EXP" ]]; then
  err "No experiment directory found in $RESULTS_DIR"; exit 2
fi

info "Latest experiment directory: $LATEST_EXP"

# Basic log validation
if ! grep -q 'Activation metrics collected' "$LATEST_EXP/log.txt"; then
  warn "No 'Activation metrics collected' line found yet. Increase steps or lower --log_interval."
else
  info "Activation collection lines detected:"; grep 'Activation metrics collected' "$LATEST_EXP/log.txt" | tail -3
fi

# Show a few activation mean metrics
info "Sample activation mean metrics:" || true
grep -E 'activation/(attn|mlp|all)/mean' "$LATEST_EXP/log.txt" | head -10 || true

#############################################
# Multi-GPU run (optional)
#############################################
if [[ $MULTI -eq 1 ]]; then
  info "Launching multi-GPU run (nproc=$NPROC)"
  FORCE_FLAG=""; [[ $FORCE_STATS -eq 1 ]] && FORCE_FLAG="--force_activation_stats"
  if [[ -n "$CONFIG_FILE" ]]; then
    cmd $RUN_TORCHRUN --nproc-per-node=$NPROC train_classifier.py \
      --config_file "$CONFIG_FILE" --log_activations $FORCE_FLAG --debug
  else
    cmd $RUN_TORCHRUN --nproc-per-node=$NPROC train_classifier.py \
      --dataset "$DATASET" --model "$MODEL" --preset "$PRESET" \
      --epochs $EPOCHS --global_batch_size $GLOBAL_BATCH \
  --log_interval $LOG_INTERVAL \
      --log_activations $FORCE_FLAG --debug "${EXTRA_ARGS[@]}"
  fi
  NEW_EXP=$(ls -td ${RESULTS_DIR}/* 2>/dev/null | head -1 || true)
  info "Multi-GPU experiment dir: $NEW_EXP"
  grep 'Activation metrics collected' "$NEW_EXP/log.txt" | tail -5 || true
fi

info "Done. Provide the last 20 lines below for feedback if needed:"
tail -n 20 "$LATEST_EXP/log.txt" || true
