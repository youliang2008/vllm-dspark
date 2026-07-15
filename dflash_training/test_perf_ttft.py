# SPDX-License-Identifier: Apache-2.0
"""TTFT / TPOP / throughput benchmark for DFlash vs AR.

Two configurations:
  1. input=4096, output=2048
  2. input=512,  output=4096

Metrics:
  TTFT  - Time To First Token (prefill latency, ms)
  TPOP  - Time Per Output Token (decode latency per token, ms)
  Decode throughput - output tokens / decode time (tok/s)
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


def make_fixed_length_prompts(
    tokenizer,
    target_input_len: int,
    num_prompts: int,
    data_path: str,
) -> list[list[int]]:
    """Build prompts with exactly target_input_len tokens each."""
    # Load raw text from dataset.
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
        # Truncate or pad to exact length.
        if len(ids) >= target_input_len:
            ids = ids[:target_input_len]
        else:
            # Repeat the text to reach target length.
            reps = (target_input_len // len(ids)) + 1
            ids = (ids * reps)[:target_input_len]
        prompts.append(ids)
    return prompts


def benchmark_run(
    args: argparse.Namespace,
    prompt_token_ids: list[list[int]],
    max_tokens: int,
    use_draft: bool,
    label: str,
) -> dict:
    """Run a single benchmark configuration."""
    print(f"\n  [{label}] Building engine...")

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

    if use_draft:
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
        }

    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    n = len(prompt_token_ids)
    prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

    # --- Warmup (5 prompts, single token) ---
    warmup_params = SamplingParams(temperature=0.0, max_tokens=1)
    warmup_prompts = prompts[: min(5, n)]
    llm.generate(warmup_prompts, warmup_params)

    # --- TTFT: measure time for 1-token generation ---
    print(f"  [{label}] Measuring TTFT ({n} prompts)...")
    ttft_params = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.monotonic()
    ttft_outputs = llm.generate(prompts, ttft_params)
    ttft_total = time.monotonic() - t0
    avg_ttft_ms = (ttft_total / n) * 1000

    # --- Full decode: measure total generation time ---
    print(f"  [{label}] Measuring decode ({n} prompts × {max_tokens} tokens)...")
    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling)
    decode_total = time.monotonic() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    # TPOP = (total_decode_time - avg_prefill_time) / output_tokens
    # Approximation: TPOP ≈ total_time / total_output_tokens (includes prefill
    # amortized over the batch, which is the realistic serving scenario).
    tpop_ms = (decode_total / total_out_tokens) * 1000 if total_out_tokens > 0 else 0
    decode_tps = total_out_tokens / decode_total if decode_total > 0 else 0

    # Spec decode metrics
    spec_metrics = {}
    if use_draft:
        metrics = llm.get_metrics()
        num_drafts = 0
        num_draft_tokens = 0
        num_accepted = 0
        for m in metrics:
            if m.name == "vllm:spec_decode_num_drafts":
                num_drafts = int(m.value)
            elif m.name == "vllm:spec_decode_num_draft_tokens":
                num_draft_tokens = int(m.value)
            elif m.name == "vllm:spec_decode_num_accepted_tokens":
                num_accepted = int(m.value)
        acc_rate = num_accepted / num_draft_tokens if num_draft_tokens > 0 else 0
        acc_len = 1 + (num_accepted / num_drafts) if num_drafts > 0 else 1.0
        spec_metrics = {
            "num_drafts": num_drafts,
            "num_draft_tokens": num_draft_tokens,
            "num_accepted": num_accepted,
            "acceptance_rate": round(acc_rate, 4),
            "acceptance_length": round(acc_len, 2),
        }

    result = {
        "label": label,
        "num_prompts": n,
        "input_tokens_per_prompt": len(prompt_token_ids[0]),
        "max_output_tokens": max_tokens,
        "actual_output_tokens": total_out_tokens,
        "avg_ttft_ms": round(avg_ttft_ms, 2),
        "tpop_ms": round(tpop_ms, 3),
        "decode_throughput_tps": round(decode_tps, 1),
        "total_decode_time_s": round(decode_total, 2),
        **spec_metrics,
    }

    print(f"  [{label}] TTFT={avg_ttft_ms:.1f}ms  "
          f"TPOP={tpop_ms:.2f}ms  "
          f"decode={decode_tps:.1f} tok/s  "
          f"out_tokens={total_out_tokens}  "
          f"time={decode_total:.1f}s")
    if spec_metrics:
        print(f"  [{label}] acceptance_rate={acc_rate:.4f}  "
              f"acceptance_length={acc_len:.2f}")

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()
    # Wait for GPU memory to be fully reclaimed.
    import time as _time
    _time.sleep(5)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--draft_model", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--num_spec_tokens", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--data_path", type=str,
                        default="train_datasets/qwen3_27b/"
                                "perfectblend_train_regen_30k.jsonl")
    parser.add_argument("--output", type=str, default="dflash_ttft_bench.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True
    )

    configs = [
        {"input_len": 4096, "output_len": 2048},
        {"input_len": 512, "output_len": 4096},
    ]

    all_results = []

    for cfg in configs:
        in_len = cfg["input_len"]
        out_len = cfg["output_len"]
        tag = f"in{in_len}_out{out_len}"

        print(f"\n{'=' * 60}")
        print(f"  Config: input={in_len}, output={out_len}")
        print(f"{'=' * 60}")

        prompt_ids = make_fixed_length_prompts(
            tokenizer, in_len, args.num_prompts, args.data_path
        )

        ar = benchmark_run(args, prompt_ids, out_len, use_draft=False,
                           label=f"AR  {tag}")
        dflash = benchmark_run(args, prompt_ids, out_len, use_draft=True,
                               label=f"DF  {tag}")

        wall_speedup = (ar["total_decode_time_s"] /
                        dflash["total_decode_time_s"]
                        if dflash["total_decode_time_s"] > 0 else 0)
        tpop_speedup = ar["tpop_ms"] / dflash["tpop_ms"] if dflash["tpop_ms"] > 0 else 0
        ttft_speedup = (ar["avg_ttft_ms"] / dflash["avg_ttft_ms"]
                        if dflash["avg_ttft_ms"] > 0 else 0)

        print(f"\n  [{tag}] Speedup — TTFT: {ttft_speedup:.2f}x  "
              f"TPOP: {tpop_speedup:.2f}x  "
              f"Decode: {wall_speedup:.2f}x")

        all_results.append({
            "config": cfg,
            "ar": ar,
            "dflash": dflash,
            "speedup": {
                "ttft": round(ttft_speedup, 3),
                "tpop": round(tpop_speedup, 3),
                "decode_throughput": round(wall_speedup, 3),
            },
        })

    print(f"\n{'=' * 60}")
    print(f"  Final Summary")
    print(f"{'=' * 60}")
    for r in all_results:
        c = r["config"]
        s = r["speedup"]
        ar = r["ar"]
        df = r["dflash"]
        print(f"\n  input={c['input_len']}, output={c['output_len']}")
        print(f"    TTFT:  AR={ar['avg_ttft_ms']:.1f}ms  "
              f"DFlash={df['avg_ttft_ms']:.1f}ms  "
              f"speedup={s['ttft']:.2f}x")
        print(f"    TPOP:  AR={ar['tpop_ms']:.2f}ms  "
              f"DFlash={df['tpop_ms']:.2f}ms  "
              f"speedup={s['tpop']:.2f}x")
        print(f"    Decode: AR={ar['decode_throughput_tps']:.1f} tok/s  "
              f"DFlash={df['decode_throughput_tps']:.1f} tok/s  "
              f"speedup={s['decode_throughput']:.2f}x")
        if "acceptance_rate" in df:
            print(f"    Accept: rate={df['acceptance_rate']:.4f}  "
                  f"length={df['acceptance_length']:.2f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[bench] saved to {args.output}")


if __name__ == "__main__":
    main()
