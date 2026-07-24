#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DSpark eval: acceptance & speedup vs AR baseline (V2 model runner).
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
# DSpark inference requires the V2 model runner.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"

TARGET_MODEL="${TARGET_MODEL:-/root/Qwen3.6-27B-FP8}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/deepspec/qwen3_27b_dspark_ckpt}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"

echo "[eval] target: ${TARGET_MODEL}"
echo "[eval] draft:  ${DRAFT_MODEL}"
echo "[eval] TP=${TENSOR_PARALLEL_SIZE}, spec_tokens=${NUM_SPEC_TOKENS}, V2 runner"

"${PY}" -m dspark_training.eval \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --num_spec_tokens "$NUM_SPEC_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"
