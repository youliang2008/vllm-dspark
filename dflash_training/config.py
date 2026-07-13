# SPDX-License-Identifier: Apache-2.0
"""Central configuration for the vLLM-native DFlash training pipeline.

All values are chosen for the Qwen3.6-27B-FP8 target
(``Qwen3_5ForConditionalGeneration``, model_type ``qwen3_5``). The draft is a
standard dense Qwen3 decoder layer (the target's exotic linear/gated attention
is NOT reproduced in the draft -- DFlash drafts are plain Qwen3 layers).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class DFlashConfig:
    # ---- Paths -------------------------------------------------------------
    target_model_path: str = _env(
        "TARGET_MODEL_PATH", "/root/Qwen3.6-27B-FP8"
    )
    train_data_path: str = _env(
        "TRAIN_DATA_PATH",
        "/root/DeepSpec/train_datasets/qwen3_27b/perfectblend_train_regen_30k.jsonl",
    )
    # Where extract.py writes per-request hidden-state shards + manifest.json.
    hidden_states_dir: str = _env(
        "HIDDEN_STATES_DIR", "/mnt/deepspec/qwen3_27b_dflash_hidden"
    )
    # Where train.py / export.py write the final DFlashDraftModel checkpoint.
    output_dir: str = _env(
        "OUTPUT_DIR", "/mnt/deepspec/qwen3_27b_dflash_ckpt"
    )

    # ---- Target dims (copied from Qwen3.6-27B-FP8 text_config) --------------
    hidden_size: int = 5120
    head_dim: int = 256
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    intermediate_size: int = 17408
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    max_position_embeddings: int = 262144
    vocab_size: int = 248320
    hidden_act: str = "silu"
    attention_bias: bool = False

    # ---- DFlash aux/target layers -----------------------------------------
    # AUX = extraction layer ids (== eagle_aux_hidden_state_layer_ids). These
    # are the target-model layers whose output hidden states feed the drafter.
    # Non-final layers, scaling the paper's 36-layer example [2,10,18,26,34]
    # to this 64-layer model. Configurable via AUX_LAYER_IDS env (comma list).
    aux_layer_ids: list[int] = field(
        default_factory=lambda: [
            int(x)
            for x in _env("AUX_LAYER_IDS", "8,20,32,44,56").split(",")
            if x.strip()
        ]
    )

    # ---- Draft architecture ------------------------------------------------
    num_draft_layers: int = int(_env("NUM_DRAFT_LAYERS", "1"))
    # DFlash non-causal cross-attention over the target-hidden context.
    causal: bool = _env("DFLASH_CAUSAL", "0") == "1"
    # Parallel-drafting mask/pard token id. Must be a reserved id that does not
    # occur as a real anchor token in training data. Its embedding is overridden
    # by a trained ``mask_embedding`` (shipped as mask_embedding.pt).
    mask_token_id: int = int(_env("MASK_TOKEN_ID", "248319"))

    # ---- Training block ----------------------------------------------------
    # Number of future tokens predicted per anchor position (draft block size).
    block_size: int = int(_env("BLOCK_SIZE", "8"))

    # ---- Semi-autoregressive Markov head (DSpark paper) --------------------
    # rank=0 disables (pure DFlash); rank=256 is the paper default.
    markov_rank: int = int(_env("MARKOV_RANK", "0"))

    # ---- Loss design -------------------------------------------------------
    # Hybrid loss: alpha_ce * CE + alpha_l1 * L1 (distribution matching).
    # Paper default: ce=0.1, l1=0.9.  Pure CE (DFlash): ce=1.0, l1=0.0.
    ce_loss_alpha: float = float(_env("CE_LOSS_ALPHA", "1.0"))
    l1_loss_alpha: float = float(_env("L1_LOSS_ALPHA", "0.0"))
    # Position exponential decay: w_k = exp(-(k-1)/gamma).
    # gamma=0 disables; paper default gamma=4.0.
    loss_decay_gamma: float = float(_env("LOSS_DECAY_GAMMA", "0.0"))

    # ---- Optimization ------------------------------------------------------
    lr: float = float(_env("LR", "1e-4"))
    min_lr: float = float(_env("MIN_LR", "1e-5"))
    weight_decay: float = float(_env("WEIGHT_DECAY", "0.0"))
    warmup_steps: int = int(_env("WARMUP_STEPS", "200"))
    max_steps: int = int(_env("MAX_STEPS", "20000"))
    grad_clip: float = float(_env("GRAD_CLIP", "1.0"))
    # Max total tokens (samples * seq_len) per no-padding length-bucketed batch.
    max_batch_tokens: int = int(_env("MAX_BATCH_TOKENS", "8192"))
    max_samples_per_batch: int = int(_env("MAX_SAMPLES_PER_BATCH", "8"))
    # Truncate very long sequences to keep memory bounded.
    max_seq_len: int = int(_env("MAX_SEQ_LEN", "4096"))
    # Separate, smaller cap used during TRAINING. The parallel-block draft
    # attention is O(L^2 * block_size) in memory, so keep this modest
    # (sequences longer than this are randomly cropped to a window).
    train_max_seq_len: int = int(_env("TRAIN_MAX_SEQ_LEN", "1024"))
    num_workers: int = int(_env("NUM_WORKERS", "4"))
    seed: int = int(_env("SEED", "0"))

    # ---- Logging / checkpointing ------------------------------------------
    log_interval: int = int(_env("LOG_INTERVAL", "10"))
    save_interval: int = int(_env("SAVE_INTERVAL", "2000"))

    @property
    def num_aux_layers(self) -> int:
        return len(self.aux_layer_ids)

    @property
    def target_layer_ids(self) -> list[int]:
        """dflash_config.target_layer_ids = aux - 1 (vLLM's i-1 convention).

        At inference vLLM collects aux layers ``[t + 1 for t in target_layer_ids]``
        which reproduces ``aux_layer_ids``.
        """
        return [a - 1 for a in self.aux_layer_ids]

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_dflash_hf_config(self) -> dict:
        """The config.json body for the exported DFlashDraftModel checkpoint."""
        return {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "num_hidden_layers": self.num_draft_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "draft_vocab_size": self.vocab_size,
            "target_hidden_size": self.hidden_size,
            "attention_bias": self.attention_bias,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
            },
            # Compat field consumed by the speculators converter path.
            "eagle_aux_hidden_state_layer_ids": list(self.aux_layer_ids),
            "dflash_config": {
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": self.target_layer_ids,
                "causal": self.causal,
                "use_aux_hidden_state": True,
                "markov_rank": self.markov_rank,
            },
        }
