#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Data preparation pipeline for Qwen3.6-27B DFlash draft model training.
#
# Three stages:
#   1. Download & split prompt data (shared across models)
#   2. Regenerate assistant answers with Qwen3.6-27B-FP8 via sglang
#   3. Precompute target hidden-state cache for training
#
# Prerequisites:
#   - pip install "sglang[all]"          (for Step 2)
#   - 8x 80GB GPUs (A100/H100)          (Step 2: 1 worker per GPU)
#   - Sufficient disk for target cache   (Step 3: scales with data size)
#
# Usage:
#   bash scripts/data/prepare_data_qwen3_27b.sh
# =============================================================================

model_path=/root/Qwen3.6-27B-FP8
config_path=config/dflash/dflash_qwen3_27b.py

dataset_name=mlabonne/open-perfectblend
test_size=0.05
train_split_path=train_datasets/perfectblend_train.jsonl
eval_data_dir=eval_datasets

train_data_path=train_datasets/qwen3_27b/perfectblend_train_regen.jsonl
cache_dir=${HOME}/.cache/deepspec/qwen3_27b_target_cache

server_host=127.0.0.1
num_workers=8
start_port=30000
concurrency=32
temperature=0.7
top_p=0.8
top_k=20
min_p=0
max_tokens=4096

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

server_addresses=()
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    server_addresses+=("${server_host}:$((start_port + worker_id))")
done

# ─── Step 1: Download and split dataset ──────────────────────────────────────
echo "Step 1/3: downloading and splitting ${dataset_name}"
python scripts/data/download_and_split.py \
    --dataset-name "${dataset_name}" \
    --test-size "${test_size}" \
    --train-output-path "${train_split_path}" \
    --test-output-dir "${eval_data_dir}" \
    --skip-existing

mkdir -p "$(dirname "${train_data_path}")"

# ─── Step 2: Regenerate answers with Qwen3.6-27B-FP8 ────────────────────────
echo "Step 2/3: generating Qwen3.6-27B train data: ${train_data_path}"
echo "Start inference server first (pick one):"
echo "  SGLang: bash scripts/data/launch_sglang_server_qwen3_27b.sh"
echo "  vLLM:   bash scripts/data/launch_vllm_server_qwen3_27b.sh"
python scripts/data/generate_train_data.py \
    --model "${model_path}" \
    --server-address "${server_addresses[@]}" \
    --concurrency "${concurrency}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${top_k}" \
    --min-p "${min_p}" \
    --max-tokens "${max_tokens}" \
    --disable-thinking \
    --resume \
    --input-file-path "${train_split_path}" \
    --output-file-path "${train_data_path}"

# ─── Step 3: Prepare target hidden-state cache ───────────────────────────────
echo "Stop sglang before Step 3 if it is using the same GPUs."
echo "Step 3/3: preparing Qwen3.6-27B target cache: ${cache_dir}"
# IMPORTANT: Qwen3.5/3.6 linear-attention (DeltaNet/Mamba-style) produces NaN on
# PADDED batches in the torch fallback path (flash-linear-attention absent). The
# script avoids padding by grouping equal-length samples into unpadded batches,
# so multi-sample batching is safe here. --local-batch-size caps samples/batch
# and --max-batch-tokens caps total tokens/batch (adaptive: big batches for
# short seqs, small for long). Lower --max-batch-tokens if you hit CUDA OOM.
#
# Launched with torchrun for 8-way DATA parallelism: each GPU holds one full
# model copy and processes 1/8 of the data (all GPUs busy). This is far faster
# than the old device_map='auto' single-process path, which was serial across
# GPUs. If a single model copy ever fails to fit on one GPU, fall back with
# PREPARE_CACHE_SINGLE_PROCESS=1 python scripts/data/prepare_target_cache.py ...
torchrun --nproc_per_node=8 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size 32 \
    --max-batch-tokens 4096 \
    --num-workers 8
