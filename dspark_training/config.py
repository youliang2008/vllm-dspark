# SPDX-License-Identifier: Apache-2.0
"""Central configuration for the self-contained DSpark training pipeline.

Produces a checkpoint natively loadable by vLLM's ``Qwen3DSparkModel`` +
``DSparkSpeculator`` (semi-autoregressive parallel drafting with a Markov head).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class DSparkConfig:
    # ---- Paths -------------------------------------------------------------
    target_model_path: str = _env("TARGET_MODEL_PATH", "/root/Qwen3.6-27B-FP8")
    train_data_path: str = _env(
        "TRAIN_DATA_PATH",
        "/root/DeepSpec/train_datasets/qwen3_27b/perfectblend_train_regen_30k.jsonl",
    )
    hidden_states_dir: str = _env(
        "HIDDEN_STATES_DIR", "/mnt/deepspec/qwen3_27b_dflash_hidden"
    )
    output_dir: str = _env("OUTPUT_DIR", "/mnt/deepspec/qwen3_27b_dspark_ckpt")

    # ---- Target dims (Qwen3.6-27B-FP8 text_config) ------------------------
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

    # ---- Aux / target layers ----------------------------------------------
    aux_layer_ids: list[int] = field(
        default_factory=lambda: [
            int(x)
            for x in _env("AUX_LAYER_IDS", "8,20,32,44,56").split(",")
            if x.strip()
        ]
    )

    # ---- Draft architecture ------------------------------------------------
    num_draft_layers: int = int(_env("NUM_DRAFT_LAYERS", "1"))
    causal: bool = _env("DSPARK_CAUSAL", "0") == "1"
    block_size: int = int(_env("BLOCK_SIZE", "8"))
    # Markov head is mandatory for DSpark (semi-autoregressive). rank>0 required.
    markov_rank: int = int(_env("MARKOV_RANK", "256"))
    # Noise/mask token id; its target embed_tokens row is used as the mask input.
    mask_token_id: int = int(_env("MASK_TOKEN_ID", "248319"))
    dspark_bonus_anchor: bool = True

    # ---- Optimization ------------------------------------------------------
    lr: float = float(_env("LR", "1e-4"))
    min_lr: float = float(_env("MIN_LR", "1e-5"))
    weight_decay: float = float(_env("WEIGHT_DECAY", "0.0"))
    warmup_steps: int = int(_env("WARMUP_STEPS", "200"))
    max_steps: int = int(_env("MAX_STEPS", "20000"))
    grad_clip: float = float(_env("GRAD_CLIP", "1.0"))
    max_batch_tokens: int = int(_env("MAX_BATCH_TOKENS", "8192"))
    max_samples_per_batch: int = int(_env("MAX_SAMPLES_PER_BATCH", "8"))
    max_seq_len: int = int(_env("MAX_SEQ_LEN", "4096"))
    train_max_seq_len: int = int(_env("TRAIN_MAX_SEQ_LEN", "1024"))
    num_workers: int = int(_env("NUM_WORKERS", "4"))
    seed: int = int(_env("SEED", "0"))
    log_interval: int = int(_env("LOG_INTERVAL", "10"))
    save_interval: int = int(_env("SAVE_INTERVAL", "2000"))

    @property
    def num_aux_layers(self) -> int:
        return len(self.aux_layer_ids)

    @property
    def target_layer_ids(self) -> list[int]:
        """DSpark indexes target layers as aux_id - 1 (matches dense configs)."""
        return [a - 1 for a in self.aux_layer_ids]

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_hf_config(self) -> dict:
        """The config.json body for the exported Qwen3DSparkModel checkpoint."""
        return {
            "architectures": ["Qwen3DSparkModel"],
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
            "eagle_aux_hidden_state_layer_ids": list(self.aux_layer_ids),
            "target_layer_ids": list(self.target_layer_ids),
            "markov_rank": self.markov_rank,
            "mask_token_id": self.mask_token_id,
            "dspark_bonus_anchor": self.dspark_bonus_anchor,
        }
