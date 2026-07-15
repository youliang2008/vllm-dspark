#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Performance benchmark: Qwen3.6-27B-FP8 with and without DFlash draft.
#
# Each (config, mode) pair runs as a SEPARATE subprocess so GPU memory
# is fully reclaimed between runs.
#
# Two configurations:
#   1. input=4096, output=2048
#   2. input=512,  output=4096
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd /tmp

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TARGET_MODEL="${TARGET_MODEL:-/root/Qwen3.6-27B-FP8}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/deepspec/qwen3_27b_dflash_ckpt}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/train_datasets/qwen3_27b/perfectblend_train_regen_30k.jsonl}"
OUTPUT="${OUTPUT:-dflash_bench_qwen3_27b.json}"

AR_FILE="/tmp/bench_ar_$$.json"
DF_FILE="/tmp/bench_df_$$.json"

echo "=========================================="
echo " DFlash Performance Benchmark"
echo "=========================================="
echo " Target:    ${TARGET_MODEL}"
echo " Draft:     ${DRAFT_MODEL}"
echo " TP:        ${TENSOR_PARALLEL_SIZE}"
echo " Prompts:   ${NUM_PROMPTS}"
echo " Context:   ${MAX_MODEL_LEN}"
echo "=========================================="

CONFIGS=("4096:2048" "512:4096")
ALL_RESULTS=""

for cfg_str in "${CONFIGS[@]}"; do
    INPUT_LEN="${cfg_str%%:*}"
    OUTPUT_LEN="${cfg_str##*:}"
    echo ""
    echo "============================================================"
    echo "  Config: input=${INPUT_LEN}, output=${OUTPUT_LEN}"
    echo "============================================================"

    # AR baseline (separate subprocess)
    echo "  [AR] Running..."
    python -m dflash_training.run_single_bench \
        --target_model "$TARGET_MODEL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --max_model_len "$MAX_MODEL_LEN" \
        --num_prompts "$NUM_PROMPTS" \
        --input_len "$INPUT_LEN" \
        --output_len "$OUTPUT_LEN" \
        --data_path "$DATA_PATH" \
        --mode ar \
        --output_file "$AR_FILE"
    echo "  [AR] Done"

    # DFlash (separate subprocess)
    echo "  [DFlash] Running..."
    python -m dflash_training.run_single_bench \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --max_model_len "$MAX_MODEL_LEN" \
        --num_prompts "$NUM_PROMPTS" \
        --input_len "$INPUT_LEN" \
        --output_len "$OUTPUT_LEN" \
        --data_path "$DATA_PATH" \
        --mode dflash \
        --output_file "$DF_FILE"
    echo "  [DFlash] Done"

    # Merge results from files
    result=$(python -c "
import json, sys
with open(sys.argv[1]) as f: ar = json.load(f)
with open(sys.argv[2]) as f: df = json.load(f)
wall = ar['total_decode_time_s'] / df['total_decode_time_s'] if df['total_decode_time_s'] > 0 else 0
tpop = ar['tpop_ms'] / df['tpop_ms'] if df['tpop_ms'] > 0 else 0
ttft = ar['avg_ttft_ms'] / df['avg_ttft_ms'] if df['avg_ttft_ms'] > 0 else 0
r = {'config': {'input_len': int(sys.argv[3]), 'output_len': int(sys.argv[4])},
     'ar': ar, 'dflash': df,
     'speedup': {'ttft': round(ttft,3), 'tpop': round(tpop,3),
                 'decode_throughput': round(wall,3)}}
print(json.dumps(r))
" "$AR_FILE" "$DF_FILE" "$INPUT_LEN" "$OUTPUT_LEN")

    ALL_RESULTS="${ALL_RESULTS}${result}"$'\n'

    # Print per-config summary
    echo "$result" | python -c "
import json, sys
r = json.loads(sys.stdin.read())
c = r['config']; s = r['speedup']; ar = r['ar']; df = r['dflash']
print(f'  input={c[\"input_len\"]}, output={c[\"output_len\"]}')
print(f'    TTFT:   AR={ar[\"avg_ttft_ms\"]:.1f}ms  DFlash={df[\"avg_ttft_ms\"]:.1f}ms  speedup={s[\"ttft\"]:.2f}x')
print(f'    TPOP:   AR={ar[\"tpop_ms\"]:.2f}ms  DFlash={df[\"tpop_ms\"]:.2f}ms  speedup={s[\"tpop\"]:.2f}x')
print(f'    Decode: AR={ar[\"decode_throughput_tps\"]:.1f} tok/s  DFlash={df[\"decode_throughput_tps\"]:.1f} tok/s  speedup={s[\"decode_throughput\"]:.2f}x')
if 'acceptance_length' in df:
    print(f'    Accept: rate={df[\"acceptance_rate\"]:.4f}  length={df[\"acceptance_length\"]:.2f}')
"
done

echo ""
echo "=========================================="
echo "  All configs complete"
echo "=========================================="

echo "$ALL_RESULTS" > "$OUTPUT"
echo "  Results saved to $OUTPUT"
rm -f "$AR_FILE" "$DF_FILE"
