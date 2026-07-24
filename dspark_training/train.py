# SPDX-License-Identifier: Apache-2.0
"""Stage 4: DDP training loop for the DSpark draft.

Mirrors the DFlash trainer: frozen, target-shared embed_tokens + lm_head; plain
DDP data parallelism; chunked semi-autoregressive CE (with teacher-forced Markov
bias). Exports a Qwen3DSparkModel checkpoint.

Launch (see scripts/train.sh):
    torchrun --nproc_per_node=8 -m dspark_training.train
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .config import DSparkConfig
from .dataset import DSparkBatchDataset
from .draft_model import DSparkDraftForTraining, dspark_block_loss
from .export import export_checkpoint

EMBED_KEY = "model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"


def _is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _setup_dist() -> tuple[int, int, int]:
    if _is_dist():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _load_frozen_tensor(cfg: DSparkConfig, key: str, device) -> torch.Tensor:
    index_path = os.path.join(cfg.target_model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        raise KeyError(f"{key} not found in target index {index_path}")
    shard = os.path.join(cfg.target_model_path, weight_map[key])
    with safe_open(shard, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    return tensor.to(device=device, dtype=torch.bfloat16)


def _lr_at(step: int, cfg: DSparkConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cos


def main() -> None:
    cfg = DSparkConfig()
    rank, world_size, local_rank = _setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(cfg.seed + rank)

    is_main = rank == 0
    if is_main:
        print(f"[train] world_size={world_size} cfg={cfg}", flush=True)

    embed_w = _load_frozen_tensor(cfg, EMBED_KEY, device).detach()
    embed_w.requires_grad_(False)
    lm_head_w = _load_frozen_tensor(cfg, LM_HEAD_KEY, device).detach()
    lm_head_w.requires_grad_(False)
    if is_main:
        print(
            f"[train] frozen embed={tuple(embed_w.shape)} "
            f"lm_head={tuple(lm_head_w.shape)}",
            flush=True,
        )

    model = DSparkDraftForTraining(cfg).to(device=device, dtype=torch.bfloat16)
    model.set_frozen_tensors(embed_w, lm_head_w)

    ddp_model = model
    if world_size > 1:
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    dataset = DSparkBatchDataset(cfg, rank=rank, world_size=world_size, seed=cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    if is_main:
        print(f"[train] rank0 batches/epoch={len(dataset)}", flush=True)

    layout_cache: dict[int, dict] = {}

    def get_layout(seq_len: int) -> dict:
        if seq_len not in layout_cache:
            layout_cache[seq_len] = model.build_block_layout(seq_len, device)
        return layout_cache[seq_len]

    step = 0
    epoch = 0
    data_iter = iter(loader)
    model.train()
    while step < cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            dataset.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        aux_hidden = batch["aux_hidden"].to(device, non_blocking=True)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue
        layout = get_layout(seq_len)

        lr = _lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        hidden = ddp_model(input_ids, aux_hidden, layout)
        loss, metrics = dspark_block_loss(
            hidden,
            model.lm_head_weight,
            input_ids,
            layout,
            markov_head=model.markov_head,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()

        if is_main and step % cfg.log_interval == 0:
            acc_str = " ".join(
                f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith("acc@")
            )
            print(
                f"[train] step={step}/{cfg.max_steps} lr={lr:.2e} "
                f"loss={metrics['loss']:.4f} acc={metrics.get('acc', 0):.3f} "
                f"L={seq_len} B={input_ids.shape[0]} {acc_str}",
                flush=True,
            )

        if is_main and step > 0 and step % cfg.save_interval == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"step_{step}")
            export_checkpoint(model, cfg, ckpt_dir)
            print(f"[train] saved checkpoint -> {ckpt_dir}", flush=True)

        step += 1

    if is_main:
        export_checkpoint(model, cfg, cfg.output_dir)
        print(f"[train] final checkpoint -> {cfg.output_dir}", flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
