#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stage 1 launcher: extract target aux hidden states (data-parallel 4 x TP2).
# DSpark reuses the same aux shards as DFlash; no target_last_hidden.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Symlink dspark_training into a clean dir so the project's vllm/ source
# tree doesn't shadow the installed vllm package.
_PY_CLEAN=$(mktemp -d)
ln -s "${PROJECT_ROOT}/dspark_training" "${_PY_CLEAN}/dspark_training"
export PYTHONPATH="${_PY_CLEAN}:${PYTHONPATH:-}"
trap 'rm -rf "${_PY_CLEAN}"' EXIT
cd /tmp

PY="${PY:-/root/anaconda3/envs/deepspec/bin/python}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
unset PYTORCH_CUDA_ALLOC_CONF

export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/root/Qwen3.6-27B-FP8}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/train_datasets/qwen3_27b/perfectblend_train_regen_30k.jsonl}"
export HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-/mnt/deepspec/qwen3_27b_dflash_hidden}"
export AUX_LAYER_IDS="${AUX_LAYER_IDS:-8,20,32,44,56}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"

NUM_SHARDS="${NUM_SHARDS:-4}"
mkdir -p "${HIDDEN_STATES_DIR}"

pids=()
for (( i=0; i<NUM_SHARDS; i++ )); do
    gpu_start=$(( i * TENSOR_PARALLEL_SIZE ))
    gpus=""
    for (( g=0; g<TENSOR_PARALLEL_SIZE; g++ )); do
        gpus="${gpus}$(( gpu_start + g )),"
    done
    gpus="${gpus%,}"
    echo "[extract.sh] worker ${i}/${NUM_SHARDS} on GPUs ${gpus}"
    CUDA_VISIBLE_DEVICES="${gpus}" \
        SHARD_INDEX="${i}" NUM_SHARDS="${NUM_SHARDS}" \
        "${PY}" -m dspark_training.extract &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "${pid}" || fail=1
done
if [[ "${fail}" -ne 0 ]]; then
    echo "[extract.sh] at least one worker failed" >&2
    exit 1
fi
echo "[extract.sh] done -> ${HIDDEN_STATES_DIR}"
