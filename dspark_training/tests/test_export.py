# SPDX-License-Identifier: Apache-2.0
import json
import os

from safetensors.torch import load_file

from dspark_training.config import DSparkConfig
from dspark_training.draft_model import DSparkDraftForTraining
from dspark_training.export import export_checkpoint


def _tiny_cfg() -> DSparkConfig:
    cfg = DSparkConfig()
    cfg.hidden_size = 64
    cfg.head_dim = 16
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    cfg.vocab_size = 100
    cfg.mask_token_id = 99
    cfg.markov_rank = 8
    cfg.num_draft_layers = 1
    return cfg


def test_export_keys_match_qwen3dspark_contract(tmp_path):
    cfg = _tiny_cfg()
    m = DSparkDraftForTraining(cfg)
    out = export_checkpoint(m, cfg, str(tmp_path))
    keys = set(load_file(os.path.join(out, "model.safetensors")).keys())
    assert "markov_head.markov_w1.weight" in keys
    assert "markov_head.markov_w2.weight" in keys
    assert "fc.weight" in keys and "norm.weight" in keys
    assert not any("embed_tokens" in k or "lm_head" in k for k in keys)
    assert not os.path.exists(os.path.join(out, "mask_embedding.pt"))
    hf = json.load(open(os.path.join(out, "config.json")))
    assert hf["architectures"] == ["Qwen3DSparkModel"]
    assert hf["dspark_bonus_anchor"] is True


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_export_keys_match_qwen3dspark_contract(Path(d))
    print("PASS test_export")
