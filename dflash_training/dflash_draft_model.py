# SPDX-License-Identifier: Apache-2.0
"""Stage 3 (model): a plain-PyTorch DFlash draft used for TRAINING.

The module's trainable ``state_dict`` keys map 1:1 to what vLLM's
``DFlashQwen3ForCausalLM.load_weights`` expects (before its ``model.`` prefixing
and ``q/k/v -> qkv`` / ``gate/up -> gate_up`` fusion):

    fc.weight
    hidden_norm.weight
    norm.weight
    layers.{i}.input_layernorm.weight
    layers.{i}.post_attention_layernorm.weight
    layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
    layers.{i}.self_attn.{q_norm,k_norm}.weight
    layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight

``embed_tokens`` and ``lm_head`` are the *target* model's (frozen, shared at
inference) and are NOT part of the saved draft. ``mask_embedding`` is trained
and shipped separately as ``mask_embedding.pt``.

Training reproduces vLLM's DFlash inference contract
(see vllm/v1/spec_decode/dflash.py, vllm/model_executor/models/qwen3_dflash.py):
  * context K/V come from ``k/v_proj(hidden_norm(fc(concat(aux_hidden))))`` with
    per-head k_norm + RoPE at the context positions;
  * query tokens are ``[anchor, mask_1..mask_N]`` per anchor position, embedded
    via the target ``embed_tokens`` (anchor) and the trained ``mask_embedding``;
  * a query block for anchor ``a`` attends to context positions ``< a`` (so it
    never sees the target hidden state of the token it must predict) plus the
    tokens within its own block; each mask slot ``k`` predicts ``x[a + k]``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DFlashConfig


# ---- Markov head (semi-autoregressive, DSpark paper) -----------------------

class VanillaMarkov(nn.Module):
    """Low-rank first-order transition bias: B(x_{k-1}, x_k) = W2(W1[x_{k-1}])."""

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def compute_bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """prev_token_ids: [...]; returns bias: [..., vocab_size]."""
        return self.markov_w2(self.markov_w1(prev_token_ids.long()))

    def apply_to_logits(
        self, base_logits: torch.Tensor, prev_token_ids: torch.Tensor
    ) -> torch.Tensor:
        return base_logits + self.compute_bias(prev_token_ids)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,  # [B, block_size, V]
        first_prev_ids: torch.Tensor,  # [B]
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Semi-autoregressive sampling with Markov corrections."""
        batch_size, block_size = base_logits.shape[:2]
        sampled = []
        corrected_logits = []
        prev = first_prev_ids.long()
        for k in range(block_size):
            step_logits = self.apply_to_logits(base_logits[:, k, :], prev)
            corrected_logits.append(step_logits.unsqueeze(1))
            if temperature > 0:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(1)
            else:
                next_tok = step_logits.argmax(-1)
            sampled.append(next_tok)
            prev = next_tok
        return torch.stack(sampled, dim=1), torch.cat(corrected_logits, dim=1)


# ---- RMSNorm ---------------------------------------------------------------

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
        return (self.weight * x.to(dtype))


def _rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard (neox) rotary cos/sin for the given 1-D positions.

    Returns cos, sin of shape [num_positions, head_dim].
    """
    device = positions.device
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [P, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [P, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x: [B, S, H, D]; cos/sin: [S, D] (neox)."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


class DraftAttention(nn.Module):
    def __init__(self, cfg: DFlashConfig):
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

    def project_context_kv(
        self,
        ctx_normed: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Context K/V from hidden_norm(fc(aux)); mirrors precompute_and_store."""
        b, s, _ = ctx_normed.shape
        k = self.k_proj(ctx_normed).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(ctx_normed).view(b, s, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = _apply_rope(k, cos, sin)
        return k, v

    def query_qkv(
        self,
        q_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = q_hidden.shape
        q = self.q_proj(q_hidden).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(q_hidden).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(q_hidden).view(b, s, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        return q, k, v

    def attend(
        self,
        q: torch.Tensor,          # [B, Q, Hq, D]
        k: torch.Tensor,          # [B, K, Hkv, D]
        v: torch.Tensor,          # [B, K, Hkv, D]
        attn_bias: torch.Tensor,  # [Q, K] additive float
    ) -> torch.Tensor:
        b, q_len, hq, d = q.shape
        # GQA: expand kv heads to query heads.
        rep = hq // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1, 2)  # [B, Hq, Q, D]
        k = k.transpose(1, 2)  # [B, Hq, K, D]
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias[None, None].to(q.dtype)
        )
        out = out.transpose(1, 2).reshape(b, q_len, hq * d)
        return self.o_proj(out)


class DraftMLP(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftDecoderLayer(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.self_attn = DraftAttention(cfg)
        self.mlp = DraftMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class DFlashDraftForTraining(nn.Module):
    """DFlash draft trained from extracted target hidden states."""

    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.fc = nn.Linear(h * cfg.num_aux_layers, h, bias=False)
        self.hidden_norm = RMSNorm(h, cfg.rms_norm_eps)
        self.norm = RMSNorm(h, cfg.rms_norm_eps)
        self.mask_embedding = nn.Parameter(torch.zeros(h))
        self.layers = nn.ModuleList(
            [DraftDecoderLayer(cfg) for _ in range(cfg.num_draft_layers)]
        )
        # Semi-autoregressive Markov head (DSpark paper).
        self.markov_head: VanillaMarkov | None = None
        if cfg.markov_rank > 0:
            self.markov_head = VanillaMarkov(cfg.vocab_size, cfg.markov_rank)
        # Frozen, target-shared tensors. Stored as PLAIN attributes (not buffers
        # or params) so they are excluded from the draft state_dict AND never
        # broadcast/sharded by DDP/FSDP. Populated via set_frozen_tensors and
        # expected to already live on the correct device.
        self.embed_tokens_weight: torch.Tensor | None = None
        self.lm_head_weight: torch.Tensor | None = None

    # ---- frozen target tensors -------------------------------------------
    def set_frozen_tensors(
        self, embed_tokens_weight: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> None:
        self.embed_tokens_weight = embed_tokens_weight
        self.lm_head_weight = lm_head_weight

    def draft_state_dict(self) -> dict[str, torch.Tensor]:
        """Trainable draft weights (incl. Markov head), excluding embed/lm_head/mask_embedding."""
        sd = {}
        for name, p in self.named_parameters():
            if name == "mask_embedding":
                continue
            sd[name] = p.detach()
        return sd

    # ---- layout / mask helpers -------------------------------------------
    def build_block_layout(
        self, seq_len: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Precompute query positions, attention mask and target gather indices
        for a sequence of length ``seq_len``. Depends only on (L, block, causal).
        """
        n = self.cfg.block_size
        bpr = n + 1  # block per row (anchor + N masks)
        a = torch.arange(seq_len, device=device)
        off = torch.arange(bpr, device=device)
        # query positions: pos[a, off] = a + off
        q_pos = (a[:, None] + off[None, :]).reshape(-1)  # [Q]
        q_anchor = (a[:, None].expand(seq_len, bpr)).reshape(-1)  # [Q] anchor id
        q_off = (off[None, :].expand(seq_len, bpr)).reshape(-1)  # [Q] offset
        q = q_pos.shape[0]

        # ---- attention bias [Q, L + Q] ----
        neg = torch.finfo(torch.float32).min
        # context part: query (anchor a) attends to context j < a
        ctx_allow = q_anchor[:, None] > a[None, :]  # [Q, L]  (j < a)
        # query self part: same anchor; full block if non-causal else causal
        same_anchor = q_anchor[:, None] == q_anchor[None, :]  # [Q, Q]
        if self.cfg.causal:
            order_ok = q_off[None, :] <= q_off[:, None]
            q_allow = same_anchor & order_ok
        else:
            q_allow = same_anchor
        allow = torch.cat([ctx_allow, q_allow], dim=1)  # [Q, L+Q]
        attn_bias = torch.where(
            allow, torch.zeros((), device=device), torch.full((), neg, device=device)
        ).float()

        # ---- targets: mask slot (off>=1) at anchor a predicts x[a+off] ----
        tgt_index = q_anchor + q_off  # [Q] absolute target position
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

    # ---- forward ----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        aux_hidden: torch.Tensor,      # [B, L, num_aux, hidden]
        layout: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns final hidden states for all query tokens: [B, Q, hidden]."""
        b, seq_len = input_ids.shape
        n1 = self.cfg.block_size + 1
        dtype = self.fc.weight.dtype

        # Context states c = fc(concat(aux)); then hidden_norm for KV proj.
        c = self.fc(aux_hidden.reshape(b, seq_len, -1).to(dtype))
        c_normed = self.hidden_norm(c)

        # RoPE tables. Context positions 0..L-1; query positions up to L-1+N.
        ctx_pos = torch.arange(seq_len, device=input_ids.device)
        ctx_cos, ctx_sin = _rope_cos_sin(
            ctx_pos, self.cfg.head_dim, self.cfg.rope_theta, dtype
        )
        q_pos = layout["q_pos"]
        q_cos, q_sin = _rope_cos_sin(
            q_pos, self.cfg.head_dim, self.cfg.rope_theta, dtype
        )

        # Query embeddings: anchor(off=0)=embed(x[a]); masks=mask_embedding.
        anchor_emb = F.embedding(input_ids, self.embed_tokens_weight.to(dtype))
        query_emb = self.mask_embedding.to(dtype).view(1, 1, 1, -1).expand(
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
        hidden = self.norm(residual)
        return hidden


def block_hybrid_loss(
    hidden: torch.Tensor,               # [B, Q, hidden]
    lm_head_weight: torch.Tensor,       # [vocab, hidden]
    input_ids: torch.Tensor,            # [B, L]
    layout: dict[str, torch.Tensor],
    *,
    markov_head: VanillaMarkov | None = None,
    target_last_hidden: torch.Tensor | None = None,  # [B, L, hidden] or None
    ce_loss_alpha: float = 1.0,
    l1_loss_alpha: float = 0.0,
    loss_decay_gamma: float = 0.0,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hybrid loss: CE + optional L1 distribution matching + position weights
    + optional Markov head semi-autoregressive corrections.

    All logit computation is chunked to bound lm_head memory.
    """
    b, q_total, h = hidden.shape
    n1 = layout["q_off"].max().item() + 1 if q_total > 0 else 1
    n1 = int(n1)
    seq_len = input_ids.shape[1]
    device = hidden.device

    valid = layout["valid"]          # [Q]
    tgt_index = layout["tgt_index"]  # [Q]
    q_off = layout["q_off"]          # [Q]  offset within block (0=anchor, 1..N=mask)

    # Reshape to block structure: [B, L, n1, hidden]
    hidden_blk = hidden.reshape(b, seq_len, n1, h)
    # Only mask positions (off >= 1), skip anchor (off=0).
    mask_hidden = hidden_blk[:, :, 1:, :]  # [B, L, N, hidden]
    mask_hidden = mask_hidden.reshape(b * seq_len * (n1 - 1), h)

    # Targets and offsets for mask positions.
    off_mask = torch.arange(1, n1, device=device)  # [N]
    tgt_pos_blk = (
        torch.arange(seq_len, device=device)[:, None] + off_mask[None, :]
    )  # [L, N]
    tgt_mask = tgt_pos_blk < seq_len  # [L, N]
    tgt_pos_safe = torch.where(tgt_mask, tgt_pos_blk, torch.zeros_like(tgt_pos_blk))
    targets_mask = torch.gather(
        input_ids, 1, tgt_pos_safe.reshape(1, -1).expand(b, -1)
    )  # [B, L*N]
    targets_flat = targets_mask.reshape(-1)
    valid_flat = tgt_mask.reshape(-1)
    off_flat = off_mask[None, :].expand(seq_len, n1 - 1).reshape(-1)
    # Previous GT token: for mask[k], prev is input_ids[a+k-1].
    prev_pos = tgt_pos_safe - 1  # [L, N]
    prev_pos_safe = torch.where(prev_pos >= 0, prev_pos, torch.zeros_like(prev_pos))
    prev_tokens = torch.gather(
        input_ids, 1, prev_pos_safe.reshape(1, -1).expand(b, -1)
    ).reshape(-1)

    idx = valid_flat.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        zero = hidden.sum() * 0.0
        return zero, {"loss": 0.0}

    sel_hidden = mask_hidden[idx]       # [n, hidden]
    sel_tgt = targets_flat[idx]         # [n]
    sel_off = off_flat[idx]             # [n]
    sel_prev = prev_tokens[idx]         # [n]
    n = sel_hidden.shape[0]

    # Position decay weights: w_k = exp(-(k-1)/gamma), k = off (1-indexed).
    if loss_decay_gamma > 0:
        pos_weights = torch.exp(-(sel_off.float() - 1.0) / loss_decay_gamma)
    else:
        pos_weights = torch.ones(n, device=device)

    lm_w = lm_head_weight.to(hidden.dtype)
    # Accumulators
    ce_num = hidden.new_zeros(())
    l1_num = hidden.new_zeros(())
    l1_den = hidden.new_zeros(())
    total_correct = hidden.new_zeros(())
    per_off_correct: dict[int, float] = {}
    per_off_count: dict[int, float] = {}

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        h_chunk = sel_hidden[start:end]               # [c, hidden]
        base_logits = F.linear(h_chunk, lm_w).float()  # [c, vocab]
        t_chunk = sel_tgt[start:end]
        o_chunk = sel_off[start:end]
        w_chunk = pos_weights[start:end]

        # ---- Markov head correction (teacher-forced) ----
        if markov_head is not None:
            prev_chunk = sel_prev[start:end]
            markov_bias = markov_head.compute_bias(prev_chunk)  # [c, vocab]
            logits = base_logits + markov_bias
        else:
            logits = base_logits

        # ---- CE loss ----
        ce_per_token = F.cross_entropy(logits, t_chunk, reduction="none")
        ce_num = ce_num + (ce_per_token * w_chunk).sum()

        # ---- L1 loss (distribution matching) ----
        if l1_loss_alpha > 0 and target_last_hidden is not None:
            # Compute target logits from stored target last hidden states.
            # We need target_last_hidden[b, tgt_pos] for each valid position.
            # Reconstruct the batch and position indices from the flat idx.
            # idx maps into (b * L * N) flattened space.
            batch_idx = idx[start:end] // (seq_len * (n1 - 1))
            pos_idx_in_seq = (idx[start:end] % (seq_len * (n1 - 1))) // (n1 - 1)
            # tgt_pos = anchor + off = pos_idx_in_seq + off_chunk
            abs_pos = pos_idx_in_seq + o_chunk.long()
            abs_pos = abs_pos.clamp(max=seq_len - 1)
            tgt_h = target_last_hidden[batch_idx, abs_pos]  # [c, hidden]
            tgt_logits = F.linear(tgt_h.to(h_chunk.dtype), lm_w).float()

            draft_probs = torch.softmax(logits, dim=-1)
            target_probs = torch.softmax(tgt_logits, dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
            l1_num = l1_num + (l1_per_token * w_chunk).sum()
            l1_den = l1_den + w_chunk.sum()

        # ---- Accuracy ----
        pred = logits.argmax(-1)
        correct = (pred == t_chunk)
        total_correct = total_correct + correct.sum()
        for o in o_chunk.unique().tolist():
            m = o_chunk == o
            per_off_correct[o] = per_off_correct.get(o, 0.0) + correct[m].sum().item()
            per_off_count[o] = per_off_count.get(o, 0.0) + int(m.sum().item())

    # Combine losses
    ce_den = pos_weights.sum()
    ce_loss = ce_num / (ce_den + 1e-8)
    l1_loss = l1_num / (l1_den + 1e-8) if l1_den > 0 else ce_loss.new_zeros(())
    total_loss = ce_loss_alpha * ce_loss + l1_loss_alpha * l1_loss

    metrics = {
        "loss": float(total_loss.item()),
        "ce_loss": float(ce_loss.item()),
        "acc": float((total_correct / n).item()),
        "num_tokens": float(n),
    }
    if l1_loss_alpha > 0:
        metrics["l1_loss"] = float(l1_loss.item())
    for o in sorted(per_off_correct):
        c = per_off_count[o]
        metrics[f"acc@{o}"] = per_off_correct[o] / c if c > 0 else 0.0
    return total_loss, metrics


# Backward-compatible alias
def block_cross_entropy(
    hidden, lm_head_weight, input_ids, layout, chunk_size=128
):
    return block_hybrid_loss(
        hidden, lm_head_weight, input_ids, layout, chunk_size=chunk_size
    )
