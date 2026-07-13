# SPDX-License-Identifier: Apache-2.0
"""Stage 1: extract target hidden states with vLLM's native extract_hidden_states.

For each conversation in the 30k jsonl we render it with the target chat
template, feed the token ids to vLLM as a prompt with ``max_tokens=1``, and let
the ``ExampleHiddenStatesConnector`` dump a ``.safetensors`` shard containing:
    hidden_states : [num_tokens, num_aux_layers, hidden_size]
    token_ids     : [num_tokens]

The aux layers are selected via ``eagle_aux_hidden_state_layer_ids`` and appear
in the stored tensor in that order.

Data-parallel over multiple GPUs is achieved by launching several processes,
each with a distinct (SHARD_INDEX, NUM_SHARDS); see scripts/extract.sh.

Usage:
    python -m dflash_training.extract
Env knobs (see config.py): TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTIL,
    MAX_EXTRACT_SAMPLES, SHARD_INDEX, NUM_SHARDS.
"""
from __future__ import annotations

import json
import math
import os

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig

from .config import DFlashConfig


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


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + eps)
    return (weight.float() * x_f).to(dtype)


def _postprocess_target_last_hidden(
    shard_paths: list[str],
    target_model_path: str,
    num_aux_layers: int,
    rms_norm_eps: float,
) -> None:
    """Extract target_last_hidden from shards that include the last decoder layer.

    The last decoder layer output is at index ``num_aux_layers`` (the extra
    layer appended beyond the standard aux layers). We apply the model's
    final RMSNorm to get the post-norm representation for L1 loss.
    """
    # Load the final RMSNorm weight from the target model.
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    norm_key = "model.language_model.norm.weight"
    if norm_key not in weight_map:
        # Try alternative naming.
        for k in weight_map:
            if k.endswith(".norm.weight") and "layer" not in k:
                norm_key = k
                break
    norm_shard = os.path.join(target_model_path, weight_map[norm_key])
    norm_weight = load_file(norm_shard, device="cpu")[norm_key]
    print(f"[postprocess] loaded norm weight: {tuple(norm_weight.shape)}", flush=True)

    for path in shard_paths:
        if not os.path.exists(path):
            continue
        data = load_file(path, device="cpu")
        hs = data["hidden_states"]  # [T, num_aux_layers + 1, hidden]
        if hs.shape[1] <= num_aux_layers:
            continue  # Already processed or wrong shape.
        # Slice: aux layers vs last decoder layer.
        aux_hs = hs[:, :num_aux_layers, :]           # [T, num_aux, hidden]
        last_hs = hs[:, num_aux_layers, :]            # [T, hidden]
        # Apply final RMSNorm.
        target_last_hidden = _rms_norm(last_hs, norm_weight, rms_norm_eps)
        # Rewrite shard with separated tensors.
        new_data = {
            "hidden_states": aux_hs.contiguous(),
            "token_ids": data["token_ids"],
            "target_last_hidden": target_last_hidden.contiguous(),
        }
        save_file(new_data, path)


def main() -> None:
    cfg = DFlashConfig()
    os.makedirs(cfg.hidden_states_dir, exist_ok=True)

    tp_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "2"))
    gpu_mem_util = float(os.environ.get("GPU_MEMORY_UTIL", "0.90"))
    max_samples = int(os.environ.get("MAX_EXTRACT_SAMPLES", "0"))  # 0 = all
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

    # Optionally extract the last decoder layer for L1 loss.
    extract_last_hidden = os.environ.get("EXTRACT_TARGET_LAST_HIDDEN", "0") == "1"
    # Read target model config to find the last decoder layer index.
    from transformers import AutoConfig as _AC
    _target_cfg = _AC.from_pretrained(cfg.target_model_path, trust_remote_code=True)
    _text_cfg = getattr(_target_cfg, "text_config", _target_cfg)
    num_target_layers = int(getattr(_text_cfg, "num_hidden_layers", 64))
    last_layer_id = num_target_layers - 1  # e.g. 63 for 64-layer model
    # Extraction layer ids: aux layers + last layer (if enabled).
    extraction_layer_ids = list(cfg.aux_layer_ids)
    if extract_last_hidden:
        if last_layer_id not in extraction_layer_ids:
            extraction_layer_ids.append(last_layer_id)
        print(
            f"[extract] extracting target_last_hidden from layer {last_layer_id}",
            flush=True,
        )

    # Tokenize each conversation and build a prompt + a deterministic save path.
    prompts: list[dict] = []
    sampling_params: list[SamplingParams] = []
    manifest_entries: list[dict] = []
    for local_idx, (row_id, conv) in enumerate(rows):
        try:
            token_ids = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            token_ids = tokenizer.encode(token_ids, add_special_tokens=False)
        except Exception as exc:  # noqa: BLE001 - skip malformed rows
            print(f"[extract] skip row {row_id}: {exc}", flush=True)
            continue
        if not token_ids:
            continue
        token_ids = token_ids[: cfg.max_seq_len - 1]
        # DFlash needs at least a few tokens to form a block prediction.
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
        enable_chunked_prefill=False,  # required by extract_hidden_states
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": list(extraction_layer_ids),
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": cfg.hidden_states_dir,
                "allow_custom_save_path": True,
                # Offline batch run: no concurrent readers, skip file locks.
                "use_synchronization_lock": False,
            },
        ),
    )

    outputs = llm.generate(prompts, sampling_params)

    # Reconcile actually-written paths from the outputs (defensive).
    written = 0
    path_by_written = []
    for out, entry in zip(outputs, manifest_entries):
        kv = out.kv_transfer_params or {}
        path = kv.get("hidden_states_path", entry["path"])
        entry["path"] = path
        if os.path.exists(path):
            written += 1
            path_by_written.append(entry)

    # Post-process: extract target_last_hidden from the extra last layer.
    if extract_last_hidden and path_by_written:
        shard_paths = [e["path"] for e in path_by_written]
        print(
            f"[extract] post-processing {len(shard_paths)} shards for "
            f"target_last_hidden (layer {last_layer_id})...",
            flush=True,
        )
        _postprocess_target_last_hidden(
            shard_paths=shard_paths,
            target_model_path=cfg.target_model_path,
            num_aux_layers=cfg.num_aux_layers,
            rms_norm_eps=cfg.rms_norm_eps,
        )
        print("[extract] post-processing done", flush=True)

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
                "has_target_last_hidden": extract_last_hidden,
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
