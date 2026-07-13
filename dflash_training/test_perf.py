# SPDX-License-Identifier: Apache-2.0
"""DFlash performance benchmark: AR baseline vs speculative decoding.

Uses the same vLLM flags the user requested:
  --enable-chunked-prefill --enable-prefix-caching --dtype=auto --quantization fp8
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


def load_prompts(path: str, n: int) -> list[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            conv = row.get("conversations", [])
            parts = []
            for turn in conv:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            if parts:
                prompts.append("\n\n".join(parts))
            if len(prompts) >= n:
                break
    return prompts


def _get_metric(metrics: list, name: str) -> float:
    for m in metrics:
        if m.name == name:
            return m.value
    return 0.0


def _get_per_pos(metrics: list, n: int) -> list[float]:
    counts = [0] * n
    num_drafts = _get_metric(metrics, "vllm:spec_decode_num_drafts")
    for m in metrics:
        if m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for i, v in enumerate(m.values):
                if i < n:
                    counts[i] = v
    if num_drafts == 0:
        return [0.0] * n
    return [c / num_drafts for c in counts]


def run_benchmark(
    args: argparse.Namespace,
    prompts: list[str],
    use_draft: bool,
) -> dict:
    label = "DFlash" if use_draft else "AR"
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

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

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # Warmup: 5 prompts.
    warmup_prompts = prompts[: min(5, len(prompts))]
    llm.generate(warmup_prompts, sampling_params)

    # Benchmark.
    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.monotonic() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / elapsed if elapsed > 0 else 0

    result = {
        "label": label,
        "time_s": round(elapsed, 2),
        "total_tokens": total_tokens,
        "tokens_per_second": round(tps, 1),
        "num_outputs": len(outputs),
    }

    if use_draft:
        metrics = llm.get_metrics()
        num_drafts = _get_metric(metrics, "vllm:spec_decode_num_drafts")
        num_draft_tokens = _get_metric(metrics, "vllm:spec_decode_num_draft_tokens")
        num_accepted = _get_metric(metrics, "vllm:spec_decode_num_accepted_tokens")
        acceptance_rate = num_accepted / num_draft_tokens if num_draft_tokens > 0 else 0
        acceptance_len = 1 + (num_accepted / num_drafts) if num_drafts > 0 else 1.0
        per_pos = _get_per_pos(metrics, args.num_spec_tokens)

        result.update({
            "num_drafts": int(num_drafts),
            "num_draft_tokens": int(num_draft_tokens),
            "num_accepted": int(num_accepted),
            "acceptance_rate": round(acceptance_rate, 4),
            "acceptance_length": round(acceptance_len, 2),
            "per_position_acceptance": [round(r, 4) for r in per_pos],
        })

        print(f"  drafts={int(num_drafts)}  draft_toks={int(num_draft_tokens)}  "
              f"accepted={int(num_accepted)}")
        print(f"  acceptance_rate={acceptance_rate:.4f}  "
              f"acceptance_length={acceptance_len:.2f}")
        per_pos_str = "  ".join(f"@{i+1}={r:.3f}" for i, r in enumerate(per_pos))
        print(f"  per_position: {per_pos_str}")

    print(f"\n  time={elapsed:.1f}s  tokens={total_tokens}  "
          f"throughput={tps:.1f} tok/s  outputs={len(outputs)}")

    del llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
    return result


def main():
    parser = argparse.ArgumentParser(description="DFlash perf benchmark")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--draft_model", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--num_spec_tokens", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--data_path", type=str,
                        default="/root/DeepSpec/train_datasets/qwen3_27b/"
                                "perfectblend_train_regen_30k.jsonl")
    parser.add_argument("--output", type=str, default="dflash_bench.json")
    args = parser.parse_args()

    prompts = load_prompts(args.data_path, args.num_prompts)
    print(f"[bench] loaded {len(prompts)} prompts")

    ar = run_benchmark(args, prompts, use_draft=False)
    dflash = run_benchmark(args, prompts, use_draft=True)

    wall_speedup = ar["time_s"] / dflash["time_s"] if dflash["time_s"] > 0 else 0
    tps_speedup = (dflash["tokens_per_second"] /
                   ar["tokens_per_second"] if ar["tokens_per_second"] > 0 else 0)

    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  AR throughput:      {ar['tokens_per_second']:.1f} tok/s")
    print(f"  DFlash throughput:  {dflash['tokens_per_second']:.1f} tok/s")
    print(f"  Wall-clock speedup: {wall_speedup:.2f}x")
    print(f"  Throughput speedup: {tps_speedup:.2f}x")
    if "acceptance_rate" in dflash:
        print(f"  Acceptance rate:    {dflash['acceptance_rate']:.4f}")
        print(f"  Acceptance length:  {dflash['acceptance_length']:.2f}")

    summary = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "num_spec_tokens": args.num_spec_tokens,
        "num_prompts": args.num_prompts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "ar": ar,
        "dflash": dflash,
        "wall_clock_speedup": round(wall_speedup, 3),
        "throughput_speedup": round(tps_speedup, 3),
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[bench] results saved to {args.output}")


if __name__ == "__main__":
    main()
