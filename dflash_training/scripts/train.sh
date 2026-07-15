#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stage 4 launcher: train the DFlash draft (DDP over 8 GPUs) and export a
# vLLM-native DFlashDraftModel checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd /tmp

PY="${PY:-/root/anaconda3/envs/deepspec/bin/python}"
TORCHRUN="${TORCHRUN:-/root/anaconda3/envs/deepspec/bin/torchrun}"

# Offline: this box has no internet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/root/Qwen3.6-27B-FP8}"
export HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-/mnt/deepspec/qwen3_27b_dflash_hidden}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/deepspec/qwen3_27b_dflash_ckpt}"
export AUX_LAYER_IDS="${AUX_LAYER_IDS:-8,20,32,44,56}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export MASK_TOKEN_ID="${MASK_TOKEN_ID:-248319}"
export TRAIN_MAX_SEQ_LEN="${TRAIN_MAX_SEQ_LEN:-256}"
export MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-1024}"
export MAX_SAMPLES_PER_BATCH="${MAX_SAMPLES_PER_BATCH:-2}"
export NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-1}"
# Semi-autoregressive Markov head (DSpark paper). 0=disabled, 256=paper default.
export MARKOV_RANK="${MARKOV_RANK:-0}"
# Loss design: alpha_ce * CE + alpha_l1 * L1. Paper default: ce=0.1, l1=0.9.
export CE_LOSS_ALPHA="${CE_LOSS_ALPHA:-1.0}"
export L1_LOSS_ALPHA="${L1_LOSS_ALPHA:-0.0}"
# Position decay gamma: w_k = exp(-(k-1)/gamma). 0=disabled, paper=4.0.
export LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-0.0}"

export LR="${LR:-1e-4}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export WARMUP_STEPS="${WARMUP_STEPS:-200}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

NPROC="${NPROC:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29533}"

mkdir -p "${OUTPUT_DIR}"

"${TORCHRUN}" \
    --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m dflash_training.train

echo "[train.sh] done -> ${OUTPUT_DIR}"
