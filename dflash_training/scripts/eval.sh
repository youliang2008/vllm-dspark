#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TARGET_MODEL="${TARGET_MODEL:-/root/Qwen3.6-27B-FP8}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/deepspec/qwen3_27b_dflash_ckpt}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"

echo "[eval] target: ${TARGET_MODEL}"
echo "[eval] draft: ${DRAFT_MODEL}"
echo "[eval] TP=${TENSOR_PARALLEL_SIZE}, spec_tokens=${NUM_SPEC_TOKENS}"
echo "[eval] samples=${NUM_SAMPLES}, max_new_tokens=${MAX_NEW_TOKENS}"

python -m dflash_training.eval_dflash \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --num_spec_tokens "$NUM_SPEC_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"
