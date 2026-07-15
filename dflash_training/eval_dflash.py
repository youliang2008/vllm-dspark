"""DFlash speculative decoding evaluation: acceptance rate & speedup."""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


def load_prompts(data_path: str, num_samples: int, max_prompt_tokens: int = 2048) -> list[str]:
    prompts = []
    with open(data_path) as f:
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
            if len(prompts) >= num_samples:
                break
    return prompts


def get_counter(metrics: list, name: str) -> float:
    for m in metrics:
        if m.name == name:
            return m.value
    return 0.0


def get_per_pos_acceptance(metrics: list, num_spec_tokens: int) -> list[float]:
    counts = [0] * num_spec_tokens
    num_drafts = get_counter(metrics, "vllm:spec_decode_num_drafts")
    for m in metrics:
        if m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for i, v in enumerate(m.values):
                if i < num_spec_tokens:
                    counts[i] = v
    if num_drafts == 0:
        return [0.0] * num_spec_tokens
    return [c / num_drafts for c in counts]


def run_ar_baseline(args, prompts: list[str]) -> dict:
    print("\n" + "=" * 60)
    print("  Autoregressive Baseline")
    print("=" * 60)

    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=False,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.monotonic() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / elapsed if elapsed > 0 else 0

    result = {
        "time_s": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_second": tps,
        "num_outputs": len(outputs),
    }
    print(f"\n[AR] time={elapsed:.1f}s  tokens={total_tokens}  "
          f"throughput={tps:.1f} tok/s  outputs={len(outputs)}")

    del llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
    return result


def run_dflash(args, prompts: list[str]) -> dict:
    print("\n" + "=" * 60)
    print("  DFlash Speculative Decoding")
    print("=" * 60)

    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=args.tensor_parallel_size,
        speculative_config={
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
        },
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=False,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.monotonic() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / elapsed if elapsed > 0 else 0

    metrics = llm.get_metrics()
    num_drafts = get_counter(metrics, "vllm:spec_decode_num_drafts")
    num_draft_tokens = get_counter(metrics, "vllm:spec_decode_num_draft_tokens")
    num_accepted = get_counter(metrics, "vllm:spec_decode_num_accepted_tokens")

    acceptance_rate = num_accepted / num_draft_tokens if num_draft_tokens > 0 else 0
    acceptance_len = 1 + (num_accepted / num_drafts) if num_drafts > 0 else 1.0
    per_pos = get_per_pos_acceptance(metrics, args.num_spec_tokens)

    result = {
        "time_s": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_second": tps,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted": num_accepted,
        "acceptance_rate": acceptance_rate,
        "acceptance_length": acceptance_len,
        "per_position_acceptance": per_pos,
        "num_outputs": len(outputs),
    }

    print(f"\n[DFlash] time={elapsed:.1f}s  tokens={total_tokens}  "
          f"throughput={tps:.1f} tok/s")
    print(f"  drafts={int(num_drafts)}  draft_toks={int(num_draft_tokens)}  "
          f"accepted={int(num_accepted)}")
    print(f"  acceptance_rate={acceptance_rate:.3f}  "
          f"acceptance_length={acceptance_len:.2f}")
    per_pos_str = "  ".join(f"@{i+1}={r:.3f}" for i, r in enumerate(per_pos))
    print(f"  per_position: {per_pos_str}")

    del llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
    return result


def main():
    parser = argparse.ArgumentParser(description="DFlash eval")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--draft_model", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--num_spec_tokens", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--data_path", type=str,
                        default="train_datasets/qwen3_27b/"
                                "perfectblend_train_regen_30k.jsonl")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    prompts = load_prompts(args.data_path, args.num_samples)
    print(f"[eval] loaded {len(prompts)} prompts")

    ar_result = run_ar_baseline(args, prompts)
    dflash_result = run_dflash(args, prompts)

    speedup = ar_result["time_s"] / dflash_result["time_s"] if dflash_result["time_s"] > 0 else 0
    tps_speedup = (dflash_result["tokens_per_second"] /
                   ar_result["tokens_per_second"] if ar_result["tokens_per_second"] > 0 else 0)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  AR throughput:     {ar_result['tokens_per_second']:.1f} tok/s")
    print(f"  DFlash throughput: {dflash_result['tokens_per_second']:.1f} tok/s")
    print(f"  Wall-clock speedup: {speedup:.2f}x")
    print(f"  Throughput speedup: {tps_speedup:.2f}x")
    print(f"  Acceptance rate:    {dflash_result['acceptance_rate']:.3f}")
    print(f"  Acceptance length:  {dflash_result['acceptance_length']:.2f}")

    summary = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "num_spec_tokens": args.num_spec_tokens,
        "num_samples": args.num_samples,
        "ar": ar_result,
        "dflash": dflash_result,
        "wall_clock_speedup": speedup,
        "throughput_speedup": tps_speedup,
    }

    output_path = args.output or f"dflash_eval_{args.num_samples}samples.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] results saved to {output_path}")


if __name__ == "__main__":
    main()
