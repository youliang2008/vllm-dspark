# SPDX-License-Identifier: Apache-2.0
"""Run a single benchmark: one config (input_len, output_len), one mode (ar/dflash).

Outputs JSON result to stdout. Designed to be called as a subprocess so GPU
memory is fully reclaimed when the process exits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def make_fixed_length_prompts(
    tokenizer,
    target_input_len: int,
    num_prompts: int,
    data_path: str,
) -> list[list[int]]:
    raw_texts = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            conv = row.get("conversations", [])
            parts = []
            for turn in conv:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role in ("system", "user"):
                    parts.append(f"{role}: {content}")
            if parts:
                raw_texts.append("\n\n".join(parts))
            if len(raw_texts) >= num_prompts * 3:
                break

    prompts = []
    for i in range(num_prompts):
        text = raw_texts[i % len(raw_texts)]
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_input_len:
            ids = ids[:target_input_len]
        else:
            reps = (target_input_len // len(ids)) + 1
            ids = (ids * reps)[:target_input_len]
        prompts.append(ids)
    return prompts


def _get_metric(metrics: list, name: str) -> float:
    for m in metrics:
        if m.name == name:
            return m.value
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--draft_model", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--input_len", type=int, required=True)
    parser.add_argument("--output_len", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["ar", "dflash"], required=True)
    parser.add_argument("--num_spec_tokens", type=int, default=8)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Write JSON result to this file instead of stdout")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True
    )
    prompt_ids = make_fixed_length_prompts(
        tokenizer, args.input_len, args.num_prompts, args.data_path
    )
    prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
    n = len(prompts)

    llm_kwargs: dict = dict(
        model=args.target_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype="auto",
        quantization="fp8",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=False,
        gpu_memory_utilization=0.90,
    )

    if args.mode == "dflash":
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
        }

    label = f"{args.mode.upper()} in{args.input_len}_out{args.output_len}"
    print(f"  [{label}] Building engine...", file=sys.stderr)
    llm = LLM(**llm_kwargs)

    # Warmup (5 prompts, 1 token)
    warmup = SamplingParams(temperature=0.0, max_tokens=1)
    llm.generate(prompts[: min(5, n)], warmup)

    # TTFT: 1-token generation
    ttft_sp = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.monotonic()
    llm.generate(prompts, ttft_sp)
    ttft_total = time.monotonic() - t0
    avg_ttft_ms = (ttft_total / n) * 1000

    # Full decode
    decode_sp = SamplingParams(temperature=0.0, max_tokens=args.output_len)
    t0 = time.monotonic()
    outputs = llm.generate(prompts, decode_sp)
    decode_total = time.monotonic() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    tpop_ms = (decode_total / total_out) * 1000 if total_out > 0 else 0
    decode_tps = total_out / decode_total if decode_total > 0 else 0

    result = {
        "label": label,
        "num_prompts": n,
        "input_tokens_per_prompt": len(prompt_ids[0]),
        "max_output_tokens": args.output_len,
        "actual_output_tokens": total_out,
        "avg_ttft_ms": round(avg_ttft_ms, 2),
        "tpop_ms": round(tpop_ms, 3),
        "decode_throughput_tps": round(decode_tps, 1),
        "total_decode_time_s": round(decode_total, 2),
    }

    if args.mode == "dflash":
        metrics = llm.get_metrics()
        nd = _get_metric(metrics, "vllm:spec_decode_num_drafts")
        ndt = _get_metric(metrics, "vllm:spec_decode_num_draft_tokens")
        na = _get_metric(metrics, "vllm:spec_decode_num_accepted_tokens")
        acc_rate = na / ndt if ndt > 0 else 0
        acc_len = 1 + (na / nd) if nd > 0 else 1.0
        result["num_drafts"] = int(nd)
        result["num_draft_tokens"] = int(ndt)
        result["num_accepted"] = int(na)
        result["acceptance_rate"] = round(acc_rate, 4)
        result["acceptance_length"] = round(acc_len, 2)

    # Output JSON to file (avoids stdout pollution from vLLM progress bars)
    json_str = json.dumps(result)
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(json_str)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
