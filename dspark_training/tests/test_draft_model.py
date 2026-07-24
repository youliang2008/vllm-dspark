# SPDX-License-Identifier: Apache-2.0
import torch
from dspark_training.config import DSparkConfig
from dspark_training.draft_model import DSparkDraftForTraining, dspark_block_loss


def _tiny_cfg() -> DSparkConfig:
    cfg = DSparkConfig()
    cfg.hidden_size = 64
    cfg.head_dim = 16
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    cfg.vocab_size = 100
    cfg.mask_token_id = 99
    cfg.block_size = 4
    cfg.markov_rank = 8
    cfg.num_draft_layers = 1
    return cfg


def test_forward_and_loss_and_markov_included():
    cfg = _tiny_cfg()
    m = DSparkDraftForTraining(cfg).to(torch.float32)
    embed = torch.randn(cfg.vocab_size, cfg.hidden_size)
    lm_head = torch.randn(cfg.vocab_size, cfg.hidden_size)
    m.set_frozen_tensors(embed, lm_head)
    b, seq = 2, 12
    ids = torch.randint(0, cfg.vocab_size - 1, (b, seq))
    aux = torch.randn(b, seq, cfg.num_aux_layers, cfg.hidden_size)
    layout = m.build_block_layout(seq, torch.device("cpu"))
    hidden = m(ids, aux, layout)
    assert hidden.shape[0] == b and hidden.shape[2] == cfg.hidden_size
    loss, metrics = dspark_block_loss(
        hidden, lm_head, ids, layout, markov_head=m.markov_head
    )
    assert loss.requires_grad and "acc@1" in metrics
    # Markov head must be part of the trainable state dict.
    sd = m.draft_state_dict()
    assert any(k.startswith("markov_head.") for k in sd)
    # No learned mask_embedding in DSpark.
    assert not any("mask_embedding" in k for k in sd)


if __name__ == "__main__":
    test_forward_and_loss_and_markov_included()
    print("PASS test_forward_and_loss_and_markov_included")
