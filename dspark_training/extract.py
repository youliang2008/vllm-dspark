# SPDX-License-Identifier: Apache-2.0
"""Stage 1: extract target hidden states with vLLM's native extract_hidden_states.

For each conversation, render it with the target chat template, feed the token
ids to vLLM as a prompt with ``max_tokens=1``, and let the
``ExampleHiddenStatesConnector`` dump a ``.safetensors`` shard containing:
    hidden_states : [num_tokens, num_aux_layers, hidden_size]
    token_ids     : [num_tokens]

DSpark uses the same aux hidden states as DFlash (concat via ``fc``), so existing
DFlash aux shards are reusable; this file exists for pipeline symmetry and does
NOT extract the last decoder layer (DSpark has no L1 loss).

Usage:
    python -m dspark_training.extract
"""
from __future__ import annotations

import json
import os

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig

from .config import DSparkConfig


def _read_conversations(path: str) -> list[tuple[int, list[dict]]]:
    rows: list[tuple[int, list[dict]]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations")
            if not conv:
                continue
            rows.append((obj.get("id", len(rows)), conv))
    return rows


def _select_shard(rows: list, shard_index: int, num_shards: int) -> list:
    if num_shards <= 1:
        return rows
    return rows[shard_index::num_shards]


def main() -> None:
    cfg = DSparkConfig()
    os.makedirs(cfg.hidden_states_dir, exist_ok=True)

    tp_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "2"))
    gpu_mem_util = float(os.environ.get("GPU_MEMORY_UTIL", "0.90"))
    max_samples = int(os.environ.get("MAX_EXTRACT_SAMPLES", "0"))
    shard_index = int(os.environ.get("SHARD_INDEX", "0"))
    num_shards = int(os.environ.get("NUM_SHARDS", "1"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.target_model_path, trust_remote_code=True
    )

    rows = _read_conversations(cfg.train_data_path)
    rows = _select_shard(rows, shard_index, num_shards)
    if max_samples > 0:
        rows = rows[:max_samples]
    print(
        f"[extract] shard {shard_index}/{num_shards}: {len(rows)} conversations "
        f"-> {cfg.hidden_states_dir}",
        flush=True,
    )

    prompts: list[dict] = []
    sampling_params: list[SamplingParams] = []
    manifest_entries: list[dict] = []
    for local_idx, (row_id, conv) in enumerate(rows):
        try:
            rendered = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        except Exception as exc:  # noqa: BLE001 - skip malformed rows
            print(f"[extract] skip row {row_id}: {exc}", flush=True)
            continue
        if not token_ids:
            continue
        token_ids = token_ids[: cfg.max_seq_len - 1]
        if len(token_ids) < (cfg.block_size + 2):
            continue

        shard_path = os.path.join(
            cfg.hidden_states_dir, f"shard_{shard_index:03d}_{local_idx:07d}.safetensors"
        )
        prompts.append({"prompt_token_ids": token_ids})
        sampling_params.append(
            SamplingParams(
                max_tokens=1,
                temperature=0.0,
                extra_args={
                    "kv_transfer_params": {
                        "hidden_states_path": shard_path,
                        "include_output_tokens": False,
                    }
                },
            )
        )
        manifest_entries.append(
            {"id": row_id, "path": shard_path, "num_tokens": len(token_ids)}
        )

    if not prompts:
        print("[extract] nothing to do", flush=True)
        return

    llm = LLM(
        model=cfg.target_model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=cfg.max_seq_len,
        enable_chunked_prefill=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": list(cfg.aux_layer_ids),
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": cfg.hidden_states_dir,
                "allow_custom_save_path": True,
                "use_synchronization_lock": False,
            },
        ),
    )

    outputs = llm.generate(prompts, sampling_params)

    written = 0
    path_by_written = []
    for out, entry in zip(outputs, manifest_entries):
        kv = out.kv_transfer_params or {}
        path = kv.get("hidden_states_path", entry["path"])
        entry["path"] = path
        if os.path.exists(path):
            written += 1
            path_by_written.append(entry)

    manifest_path = os.path.join(
        cfg.hidden_states_dir, f"manifest_{shard_index:03d}.json"
    )
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "target_model": cfg.target_model_path,
                "aux_layer_ids": list(cfg.aux_layer_ids),
                "hidden_size": cfg.hidden_size,
                "num_aux_layers": cfg.num_aux_layers,
                "has_target_last_hidden": False,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "entries": path_by_written,
            },
            f,
        )
    print(
        f"[extract] wrote {written}/{len(manifest_entries)} shards; "
        f"manifest -> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
