#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stage 1 launcher: extract target hidden states with vLLM's native
# extract_hidden_states, data-parallel across the 8 GPUs as 4 x TP2 groups.
#
# The 27B FP8 target (~28.5GB) is memory-tight on a single 32GB 5090, so each
# data-parallel worker uses tensor_parallel_size=2 (a 2-GPU group).
set -euo pipefail

cd "$(dirname "$0")/../.."   # -> /root/vllm (so `-m dflash_training...` resolves)

PY="${PY:-/root/anaconda3/envs/deepspec/bin/python}"

# Offline: this box has no internet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
unset PYTORCH_CUDA_ALLOC_CONF

export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/root/Qwen3.6-27B-FP8}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/root/DeepSpec/train_datasets/qwen3_27b/perfectblend_train_regen_30k.jsonl}"
export HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-/mnt/deepspec/qwen3_27b_dflash_hidden}"
export AUX_LAYER_IDS="${AUX_LAYER_IDS:-8,20,32,44,56}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
# Extract target last decoder layer hidden states for L1 loss.
# Set to 1 to enable (requires re-extraction). Adds target_last_hidden
# to each shard, enabling L1 distribution matching during training.
export EXTRACT_TARGET_LAST_HIDDEN="${EXTRACT_TARGET_LAST_HIDDEN:-0}"

# Number of data-parallel workers (each uses TENSOR_PARALLEL_SIZE GPUs).
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
        "${PY}" -m dflash_training.extract &
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
