# SPDX-License-Identifier: Apache-2.0
"""Export a checkpoint natively loadable by vLLM's Qwen3DSparkModel.

Writes config.json (architectures=["Qwen3DSparkModel"]) and model.safetensors
containing the backbone AND the Markov head. embed_tokens/lm_head are omitted
(shared from the target by load_dspark_model). No mask_embedding.pt: DSpark
masks via the noise-token vocab row.
"""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

from .config import DSparkConfig
from .draft_model import DSparkDraftForTraining


def export_checkpoint(
    model: DSparkDraftForTraining,
    cfg: DSparkConfig,
    output_dir: str | None = None,
) -> str:
    output_dir = output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg.to_hf_config(), f, indent=2)
    state = model.draft_state_dict()
    tensors = {
        k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in state.items()
    }
    save_file(tensors, os.path.join(output_dir, "model.safetensors"))
    return output_dir
