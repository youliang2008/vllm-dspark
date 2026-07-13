# SPDX-License-Identifier: Apache-2.0
"""Task 6 validation: confirm an exported checkpoint satisfies vLLM's
DFlashDraftModel contract WITHOUT loading the 27B target.

Checks:
  1. config.json has architectures=["DFlashDraftModel"] + required dflash_config.
  2. model.safetensors keys, after applying vLLM's rename (`model.` prefix) and
     q/k/v -> qkv_proj / gate/up -> gate_up_proj fusion, are all valid DFlash
     params, and every REQUIRED draft param is present.
  3. Shapes match the config dims.
  4. mask_embedding.pt is present with the right shape + mask_token_id.

SMOKE=1 first builds a TINY draft on CPU, runs a forward+loss, exports it, then
validates that export (exercises the whole model/loss/export path).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import torch
from safetensors.torch import load_file

from .config import DFlashConfig


def _expected_shapes(cfg: DFlashConfig) -> dict[str, tuple[int, ...]]:
    h = cfg.hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        "fc.weight": (h, h * cfg.num_aux_layers),
        "hidden_norm.weight": (h,),
        "norm.weight": (h,),
    }
    for i in range(cfg.num_draft_layers):
        p = f"layers.{i}."
        shapes[p + "self_attn.q_proj.weight"] = (cfg.q_size, h)
        shapes[p + "self_attn.k_proj.weight"] = (cfg.kv_size, h)
        shapes[p + "self_attn.v_proj.weight"] = (cfg.kv_size, h)
        shapes[p + "self_attn.o_proj.weight"] = (h, cfg.q_size)
        shapes[p + "self_attn.q_norm.weight"] = (cfg.head_dim,)
        shapes[p + "self_attn.k_norm.weight"] = (cfg.head_dim,)
        shapes[p + "mlp.gate_proj.weight"] = (cfg.intermediate_size, h)
        shapes[p + "mlp.up_proj.weight"] = (cfg.intermediate_size, h)
        shapes[p + "mlp.down_proj.weight"] = (h, cfg.intermediate_size)
        shapes[p + "input_layernorm.weight"] = (h,)
        shapes[p + "post_attention_layernorm.weight"] = (h,)
    return shapes


def _vllm_transform(name: str) -> str:
    """Replicate DFlashQwen3ForCausalLM.load_weights renaming + fusion."""
    if "midlayer." in name:
        name = name.replace("midlayer.", "layers.0.")
    if "lm_head" not in name and "d2t" not in name:
        name = "model." + name
    for src, dst in (
        (".q_proj", ".qkv_proj"),
        (".k_proj", ".qkv_proj"),
        (".v_proj", ".qkv_proj"),
        (".gate_proj", ".gate_up_proj"),
        (".up_proj", ".gate_up_proj"),
    ):
        if src in name:
            name = name.replace(src, dst)
            break
    return name


def _expected_vllm_params(cfg: DFlashConfig) -> set[str]:
    params = {"model.fc.weight", "model.hidden_norm.weight", "model.norm.weight",
              "model.mask_embedding", "model.embed_tokens.weight", "lm_head.weight"}
    for i in range(cfg.num_draft_layers):
        p = f"model.layers.{i}."
        params.update({
            p + "self_attn.qkv_proj.weight",
            p + "self_attn.o_proj.weight",
            p + "self_attn.q_norm.weight",
            p + "self_attn.k_norm.weight",
            p + "mlp.gate_up_proj.weight",
            p + "mlp.down_proj.weight",
            p + "input_layernorm.weight",
            p + "post_attention_layernorm.weight",
        })
    return params


def validate(ckpt_dir: str, cfg: DFlashConfig) -> list[str]:
    errors: list[str] = []

    # 1. config.json
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        return [f"missing {cfg_path}"]
    with open(cfg_path) as f:
        hf = json.load(f)
    if hf.get("architectures") != ["DFlashDraftModel"]:
        errors.append(f"architectures != ['DFlashDraftModel']: {hf.get('architectures')}")
    dfc = hf.get("dflash_config", {})
    for req in ("mask_token_id", "target_layer_ids"):
        if req not in dfc:
            errors.append(f"dflash_config missing {req}")
    if "eagle_aux_hidden_state_layer_ids" not in hf:
        errors.append("config missing eagle_aux_hidden_state_layer_ids")
    # target_layer_ids should be aux - 1
    if dfc.get("target_layer_ids") != [a - 1 for a in cfg.aux_layer_ids]:
        errors.append(
            f"target_layer_ids {dfc.get('target_layer_ids')} != aux-1 "
            f"{[a - 1 for a in cfg.aux_layer_ids]}"
        )

    # 2/3. weights + shapes
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(st_path):
        return errors + [f"missing {st_path}"]
    tensors = load_file(st_path, device="cpu")
    expected_shapes = _expected_shapes(cfg)
    expected_params = _expected_vllm_params(cfg)

    for name in tensors:
        transformed = _vllm_transform(name)
        if transformed not in expected_params:
            errors.append(f"shipped key {name} -> {transformed} not a DFlash param")
    for req, shape in expected_shapes.items():
        if req not in tensors:
            errors.append(f"missing required weight {req}")
        elif tuple(tensors[req].shape) != shape:
            errors.append(
                f"shape mismatch {req}: got {tuple(tensors[req].shape)} want {shape}"
            )

    # 4. mask_embedding.pt
    me_path = os.path.join(ckpt_dir, "mask_embedding.pt")
    if not os.path.exists(me_path):
        errors.append(f"missing {me_path}")
    else:
        state = torch.load(me_path, weights_only=True)
        if state.get("mask_token_id") != cfg.mask_token_id:
            errors.append(
                f"mask_embedding.pt mask_token_id {state.get('mask_token_id')} "
                f"!= cfg {cfg.mask_token_id}"
            )
        emb = state.get("embedding")
        if emb is None or tuple(emb.shape) != (cfg.hidden_size,):
            errors.append(f"mask_embedding shape {None if emb is None else tuple(emb.shape)}")

    return errors


def _smoke() -> str:
    """Build a tiny draft, run forward+loss, export to a temp dir, return it."""
    from .dflash_draft_model import DFlashDraftForTraining, block_cross_entropy

    cfg = DFlashConfig()
    # Shrink to CPU-friendly dims (keep num_aux from aux_layer_ids).
    cfg.hidden_size = 64
    cfg.head_dim = 16
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    cfg.vocab_size = 256
    cfg.mask_token_id = 255
    cfg.block_size = 4
    cfg.num_draft_layers = 1

    torch.manual_seed(0)
    model = DFlashDraftForTraining(cfg).to(torch.float32)
    embed = torch.randn(cfg.vocab_size, cfg.hidden_size)
    lm_head = torch.randn(cfg.vocab_size, cfg.hidden_size)
    model.set_frozen_tensors(embed, lm_head)

    b, seq_len = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size - 1, (b, seq_len))
    aux_hidden = torch.randn(b, seq_len, cfg.num_aux_layers, cfg.hidden_size)
    layout = model.build_block_layout(seq_len, torch.device("cpu"))
    hidden = model(input_ids, aux_hidden, layout)
    assert hidden.shape == (b, seq_len * (cfg.block_size + 1), cfg.hidden_size), (
        hidden.shape
    )
    loss, metrics = block_cross_entropy(hidden, lm_head, input_ids, layout)
    loss.backward()
    grad_ok = all(
        p.grad is not None for n, p in model.named_parameters() if n != "mask_embedding"
    )
    print(f"[smoke] loss={metrics['loss']:.4f} acc={metrics.get('acc', 0):.3f} "
          f"num_tokens={metrics.get('num_tokens')} grads_ok={grad_ok}")
    assert model.mask_embedding.grad is not None, "mask_embedding got no grad"

    from .export import export_checkpoint

    tmp = tempfile.mkdtemp(prefix="dflash_smoke_")
    export_checkpoint(model, cfg, tmp)
    # Validate the tiny export with the SAME tiny cfg.
    errs = validate(tmp, cfg)
    if errs:
        print("[smoke] VALIDATION ERRORS:")
        for e in errs:
            print("  -", e)
        raise SystemExit(1)
    print(f"[smoke] tiny export validated OK -> {tmp}")
    return tmp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=None)
    args = parser.parse_args()

    if os.environ.get("SMOKE", "0") == "1":
        _smoke()
        return

    cfg = DFlashConfig()
    ckpt_dir = args.ckpt_dir or cfg.output_dir
    errs = validate(ckpt_dir, cfg)
    if errs:
        print(f"[validate] {ckpt_dir}: FAILED")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"[validate] {ckpt_dir}: OK (vLLM DFlashDraftModel contract satisfied)")


if __name__ == "__main__":
    main()
