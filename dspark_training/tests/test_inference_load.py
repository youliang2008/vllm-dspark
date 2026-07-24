# SPDX-License-Identifier: Apache-2.0
"""Guards the export -> inference weight-name contract without a GPU.

Qwen3DSparkForCausalLM.load_weights prepends 'model.' to non-lm_head/embed names
and expects the Markov head as model.markov_head.markov_w1/markov_w2. The
backbone q/k/v and gate/up are fused via the inherited Qwen3 packed_modules_mapping
(q_proj/k_proj/v_proj -> qkv_proj, gate_proj/up_proj -> gate_up_proj), so the
exported separate projections are the correct on-disk form. This test asserts the
exported key set matches that contract.
"""
import os

from safetensors.torch import load_file

from dspark_training.config import DSparkConfig
from dspark_training.draft_model import DSparkDraftForTraining
from dspark_training.export import export_checkpoint


def test_markov_and_backbone_names(tmp_path):
    cfg = DSparkConfig()
    cfg.hidden_size = 64
    cfg.head_dim = 16
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    cfg.vocab_size = 100
    cfg.markov_rank = 8
    cfg.num_draft_layers = 1
    out = export_checkpoint(DSparkDraftForTraining(cfg), cfg, str(tmp_path))
    keys = set(load_file(os.path.join(out, "model.safetensors")).keys())
    assert {"markov_head.markov_w1.weight", "markov_head.markov_w2.weight"} <= keys
    assert {
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
    } <= keys


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_markov_and_backbone_names(Path(d))
    print("PASS test_inference_load")
