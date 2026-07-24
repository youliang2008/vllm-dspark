# SPDX-License-Identifier: Apache-2.0
from dspark_training.config import DSparkConfig


def test_to_hf_config_is_dspark_native():
    cfg = DSparkConfig()
    hf = cfg.to_hf_config()
    assert hf["architectures"] == ["Qwen3DSparkModel"]
    assert hf["dspark_bonus_anchor"] is True
    assert hf["markov_rank"] == cfg.markov_rank and cfg.markov_rank > 0
    assert hf["mask_token_id"] == cfg.mask_token_id
    assert hf["draft_vocab_size"] == cfg.vocab_size
    assert hf["eagle_aux_hidden_state_layer_ids"] == cfg.aux_layer_ids
    assert hf["target_layer_ids"] == [a - 1 for a in cfg.aux_layer_ids]
    # DSpark must NOT carry DFlash-only mask_embedding semantics.
    assert "dflash_config" not in hf
