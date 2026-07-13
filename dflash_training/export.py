# SPDX-License-Identifier: Apache-2.0
"""Stage 5: export a checkpoint natively loadable by vLLM's DFlashDraftModel.

Writes into ``output_dir``:
    config.json         - architectures=["DFlashDraftModel"] + dflash_config
    model.safetensors   - draft weights ONLY (fc, hidden_norm, norm, layers.*)
    mask_embedding.pt   - {"mask_token_id", "embedding"} (loaded by vLLM's
                          DFlashQwen3ForCausalLM._read_mask_embedding)

``embed_tokens`` and ``lm_head`` are intentionally omitted so vLLM shares the
target model's tensors (see vllm/.../dflash/utils.py:load_dflash_model). vLLM's
loader fuses our separate q/k/v and gate/up projections into qkv/gate_up.
"""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

from .config import DFlashConfig
from .dflash_draft_model import DFlashDraftForTraining


def export_checkpoint(
    model: DFlashDraftForTraining,
    cfg: DFlashConfig,
    output_dir: str | None = None,
) -> str:
    output_dir = output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. config.json
    hf_config = cfg.to_dflash_hf_config()
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    # 2. draft weights (bf16, cpu, contiguous) -> model.safetensors
    state = model.draft_state_dict()
    tensors = {
        k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in state.items()
    }
    save_file(tensors, os.path.join(output_dir, "model.safetensors"))

    # 3. trained mask embedding -> mask_embedding.pt
    mask_emb = model.mask_embedding.detach().to(torch.bfloat16).cpu().contiguous()
    torch.save(
        {"mask_token_id": cfg.mask_token_id, "embedding": mask_emb},
        os.path.join(output_dir, "mask_embedding.pt"),
    )

    # 4. Markov head weights (saved separately; vLLM DFlash inference
    #    does not use them, but they can be loaded for semi-autoregressive
    #    sampling via the VanillaMarkov class).
    if model.markov_head is not None:
        markov_state = {
            k: v.detach().to(torch.bfloat16).cpu().contiguous()
            for k, v in model.markov_head.named_parameters()
        }
        torch.save(markov_state, os.path.join(output_dir, "markov_head.pt"))

    return output_dir
