#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Data preparation pipeline for Qwen3.6-27B DFlash draft model training.
#
# Two stages (dflash uses its own extract.py for hidden states, not Step 3):
#   1. Download & split prompt data (shared across models)
#   2. Regenerate assistant answers with Qwen3.6-27B-FP8 via sglang
#
# After this, run dflash_training/scripts/extract.sh to extract hidden states.
#
# Prerequisites:
#   - pip install "sglang[all]"          (for Step 2)
#   - 8x 80GB GPUs (A100/H100)          (Step 2: 1 worker per GPU)
#
# Usage:
#   bash dflash_training/scripts/data/prepare_data_qwen3_27b.sh
# =============================================================================

model_path=/root/Qwen3.6-27B-FP8

dataset_name=mlabonne/open-perfectblend
test_size=0.05
train_split_path=train_datasets/perfectblend_train.jsonl
eval_data_dir=eval_datasets

train_data_path=train_datasets/qwen3_27b/perfectblend_train_regen.jsonl

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

server_addresses=()
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    server_addresses+=("${server_host}:$((start_port + worker_id))")
done

# ─── Step 1: Download and split dataset ──────────────────────────────────────
echo "Step 1/2: downloading and splitting ${dataset_name}"
python dflash_training/scripts/data/download_and_split.py \
    --dataset-name "${dataset_name}" \
    --test-size "${test_size}" \
    --train-output-path "${train_split_path}" \
    --test-output-dir "${eval_data_dir}" \
    --skip-existing

mkdir -p "$(dirname "${train_data_path}")"

# ─── Step 2: Regenerate answers with Qwen3.6-27B-FP8 ────────────────────────
echo "Step 2/2: generating Qwen3.6-27B train data: ${train_data_path}"
echo "Start inference server first (pick one):"
echo "  SGLang: bash dflash_training/scripts/data/launch_sglang_server_qwen3_27b.sh"
echo "  vLLM:   bash dflash_training/scripts/data/launch_vllm_server_qwen3_27b.sh"
python dflash_training/scripts/data/generate_train_data.py \
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

echo ""
echo "Done. Next step: extract hidden states with dflash_training/scripts/extract.sh"
