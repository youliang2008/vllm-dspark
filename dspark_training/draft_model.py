# SPDX-License-Identifier: Apache-2.0
"""DSpark draft used for TRAINING (self-contained, no dflash_training imports).

DSpark = DFlash parallel backbone + a lightweight sequential Markov head that
injects intra-block token dependency (semi-autoregressive). Two differences from
the DFlash training draft make the checkpoint load natively into vLLM's
``Qwen3DSparkModel`` / ``DSparkSpeculator``:

  * mask slots are embedded via the TARGET ``embed_tokens`` row of
    ``mask_token_id`` (a noise token), NOT a learned ``mask_embedding`` vector;
  * a trained ``markov_head`` is part of the model (exported as
    ``markov_head.markov_w1/markov_w2``) and applied left-to-right at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DSparkConfig


class VanillaMarkov(nn.Module):
    """Low-rank first-order transition bias: B(x_{k-1}) = W2(W1[x_{k-1}])."""

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def compute_bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(prev_token_ids.long()))


class RMSNorm(nn.Module):
    """RMSNorm matching vLLM's semantics (variance in fp32, weight multiply)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(dtype)


def _rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    device = positions.device
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


class DraftAttention(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_size = cfg.q_size
        self.kv_size = cfg.kv_size
        self.scaling = self.head_dim ** -0.5
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, self.q_size, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_size, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_size, bias=bias)
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def project_context_kv(self, ctx_normed, cos, sin):
        b, s, _ = ctx_normed.shape
        k = self.k_proj(ctx_normed).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(ctx_normed).view(b, s, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = _apply_rope(k, cos, sin)
        return k, v

    def query_qkv(self, q_hidden, cos, sin):
        b, s, _ = q_hidden.shape
        q = self.q_proj(q_hidden).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(q_hidden).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(q_hidden).view(b, s, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        return q, k, v

    def attend(self, q, k, v, attn_bias):
        b, q_len, hq, d = q.shape
        rep = hq // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias[None, None].to(q.dtype)
        )
        out = out.transpose(1, 2).reshape(b, q_len, hq * d)
        return self.o_proj(out)


class DraftMLP(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftDecoderLayer(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.self_attn = DraftAttention(cfg)
        self.mlp = DraftMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class DSparkDraftForTraining(nn.Module):
    """DSpark draft: DFlash backbone + noise-token masks + Markov head."""

    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.fc = nn.Linear(h * cfg.num_aux_layers, h, bias=False)
        self.hidden_norm = RMSNorm(h, cfg.rms_norm_eps)
        self.norm = RMSNorm(h, cfg.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DraftDecoderLayer(cfg) for _ in range(cfg.num_draft_layers)]
        )
        if cfg.markov_rank <= 0:
            raise ValueError("DSpark requires markov_rank > 0")
        self.markov_head = VanillaMarkov(cfg.vocab_size, cfg.markov_rank)
        self.embed_tokens_weight: torch.Tensor | None = None
        self.lm_head_weight: torch.Tensor | None = None

    def set_frozen_tensors(self, embed_tokens_weight, lm_head_weight) -> None:
        self.embed_tokens_weight = embed_tokens_weight
        self.lm_head_weight = lm_head_weight

    def draft_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: p.detach() for name, p in self.named_parameters()}

    def build_block_layout(self, seq_len: int, device: torch.device) -> dict:
        n = self.cfg.block_size
        bpr = n + 1
        a = torch.arange(seq_len, device=device)
        off = torch.arange(bpr, device=device)
        q_pos = (a[:, None] + off[None, :]).reshape(-1)
        q_anchor = (a[:, None].expand(seq_len, bpr)).reshape(-1)
        q_off = (off[None, :].expand(seq_len, bpr)).reshape(-1)
        q = q_pos.shape[0]

        neg = torch.finfo(torch.float32).min
        ctx_allow = q_anchor[:, None] > a[None, :]
        same_anchor = q_anchor[:, None] == q_anchor[None, :]
        if self.cfg.causal:
            order_ok = q_off[None, :] <= q_off[:, None]
            q_allow = same_anchor & order_ok
        else:
            q_allow = same_anchor
        allow = torch.cat([ctx_allow, q_allow], dim=1)
        attn_bias = torch.where(
            allow, torch.zeros((), device=device), torch.full((), neg, device=device)
        ).float()

        tgt_index = q_anchor + q_off
        valid = (q_off >= 1) & (tgt_index < seq_len)
        tgt_index = torch.where(valid, tgt_index, torch.zeros_like(tgt_index))
        return {
            "q_pos": q_pos,
            "q_off": q_off,
            "tgt_index": tgt_index,
            "valid": valid,
            "attn_bias": attn_bias,
            "num_query": torch.tensor(q, device=device),
        }

    def forward(self, input_ids, aux_hidden, layout) -> torch.Tensor:
        b, seq_len = input_ids.shape
        n1 = self.cfg.block_size + 1
        dtype = self.fc.weight.dtype

        c = self.fc(aux_hidden.reshape(b, seq_len, -1).to(dtype))
        c_normed = self.hidden_norm(c)

        ctx_pos = torch.arange(seq_len, device=input_ids.device)
        ctx_cos, ctx_sin = _rope_cos_sin(
            ctx_pos, self.cfg.head_dim, self.cfg.rope_theta, dtype
        )
        q_pos = layout["q_pos"]
        q_cos, q_sin = _rope_cos_sin(
            q_pos, self.cfg.head_dim, self.cfg.rope_theta, dtype
        )

        embed = self.embed_tokens_weight.to(dtype)
        anchor_emb = F.embedding(input_ids, embed)
        # DSpark: mask slots use the target embed row of mask_token_id (noise
        # token), NOT a learned mask_embedding.
        mask_vec = embed[self.cfg.mask_token_id]
        query_emb = mask_vec.view(1, 1, 1, -1).expand(
            b, seq_len, n1, self.cfg.hidden_size
        ).clone()
        query_emb[:, :, 0, :] = anchor_emb
        query_emb = query_emb.reshape(b, seq_len * n1, self.cfg.hidden_size)

        attn_bias = layout["attn_bias"]
        hidden = query_emb
        residual = None
        for layer in self.layers:
            attn = layer.self_attn
            k_ctx, v_ctx = attn.project_context_kv(c_normed, ctx_cos, ctx_sin)
            if residual is None:
                residual = hidden
                normed = layer.input_layernorm(hidden)
            else:
                residual = residual + hidden
                normed = layer.input_layernorm(residual)
            q, k_q, v_q = attn.query_qkv(normed, q_cos, q_sin)
            k = torch.cat([k_ctx, k_q], dim=1)
            v = torch.cat([v_ctx, v_q], dim=1)
            attn_out = attn.attend(q, k, v, attn_bias)
            residual = residual + attn_out
            normed = layer.post_attention_layernorm(residual)
            hidden = layer.mlp(normed)
        residual = residual + hidden
        return self.norm(residual)


def dspark_block_loss(
    hidden,
    lm_head_weight,
    input_ids,
    layout,
    *,
    markov_head: VanillaMarkov,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Semi-autoregressive CE with teacher-forced Markov bias (no L1)."""
    b, q_total, h = hidden.shape
    n1 = int(layout["q_off"].max().item() + 1) if q_total > 0 else 1
    seq_len = input_ids.shape[1]
    device = hidden.device

    hidden_blk = hidden.reshape(b, seq_len, n1, h)
    mask_hidden = hidden_blk[:, :, 1:, :].reshape(b * seq_len * (n1 - 1), h)

    off_mask = torch.arange(1, n1, device=device)
    tgt_pos_blk = torch.arange(seq_len, device=device)[:, None] + off_mask[None, :]
    tgt_mask = tgt_pos_blk < seq_len
    tgt_pos_safe = torch.where(tgt_mask, tgt_pos_blk, torch.zeros_like(tgt_pos_blk))
    targets_mask = torch.gather(
        input_ids, 1, tgt_pos_safe.reshape(1, -1).expand(b, -1)
    )
    targets_flat = targets_mask.reshape(-1)
    valid_flat = tgt_mask.reshape(-1)
    off_flat = off_mask[None, :].expand(seq_len, n1 - 1).reshape(-1)
    prev_pos = tgt_pos_safe - 1
    prev_pos_safe = torch.where(prev_pos >= 0, prev_pos, torch.zeros_like(prev_pos))
    prev_tokens = torch.gather(
        input_ids, 1, prev_pos_safe.reshape(1, -1).expand(b, -1)
    ).reshape(-1)

    idx = valid_flat.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        zero = hidden.sum() * 0.0
        return zero, {"loss": 0.0}

    sel_hidden = mask_hidden[idx]
    sel_tgt = targets_flat[idx]
    sel_off = off_flat[idx]
    sel_prev = prev_tokens[idx]
    n = sel_hidden.shape[0]

    lm_w = lm_head_weight.to(hidden.dtype)
    ce_num = hidden.new_zeros(())
    total_correct = hidden.new_zeros(())
    per_off_correct: dict[int, float] = {}
    per_off_count: dict[int, float] = {}

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        h_chunk = sel_hidden[start:end]
        base_logits = F.linear(h_chunk, lm_w).float()
        t_chunk = sel_tgt[start:end]
        o_chunk = sel_off[start:end]
        prev_chunk = sel_prev[start:end]

        logits = base_logits + markov_head.compute_bias(prev_chunk)

        ce_per_token = F.cross_entropy(logits, t_chunk, reduction="none")
        ce_num = ce_num + ce_per_token.sum()

        pred = logits.argmax(-1)
        correct = pred == t_chunk
        total_correct = total_correct + correct.sum()
        for o in o_chunk.unique().tolist():
            m = o_chunk == o
            per_off_correct[o] = per_off_correct.get(o, 0.0) + correct[m].sum().item()
            per_off_count[o] = per_off_count.get(o, 0.0) + int(m.sum().item())

    ce_loss = ce_num / (n + 1e-8)
    metrics = {
        "loss": float(ce_loss.item()),
        "ce_loss": float(ce_loss.item()),
        "acc": float((total_correct / n).item()),
        "num_tokens": float(n),
    }
    for o in sorted(per_off_correct):
        c = per_off_count[o]
        metrics[f"acc@{o}"] = per_off_correct[o] / c if c > 0 else 0.0
    return ce_loss, metrics
