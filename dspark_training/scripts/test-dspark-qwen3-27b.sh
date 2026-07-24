#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Integration smoke test: load the DSpark checkpoint under the V2 runner and
# print per-position acceptance on a small sample. Compares against the 50k CE
# DFlash baseline (acc@1 56.9% @2 28.6% @3 14.0% @4 6.1%).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TARGET_MODEL="${TARGET_MODEL:-/root/Qwen3.6-27B-FP8}" \
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/deepspec/qwen3_27b_dspark_ckpt}" \
NUM_SAMPLES="${NUM_SAMPLES:-20}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}" \
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}" \
    bash "${SCRIPT_DIR}/eval.sh"

echo "[test-dspark] done. Compare per-position acc@1..4 to the 50k CE baseline."
