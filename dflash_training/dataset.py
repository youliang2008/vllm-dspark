# SPDX-License-Identifier: Apache-2.0
"""Stage 2 (data): read extracted hidden-state shards into training batches.

Each shard (written by extract.py via vLLM's ExampleHiddenStatesConnector) holds:
    hidden_states : [num_tokens, num_aux_layers, hidden_size]
    token_ids     : [num_tokens]

We form NO-PADDING, equal-length batches (length-bucketing) to avoid any padding
in the parallel-block draft attention. Long sequences are randomly cropped to
``train_max_seq_len``; short sequences keep their true length and bucket with
other sequences of the same length.
"""
from __future__ import annotations

import glob
import json
import os
import random

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from .config import DFlashConfig


def load_manifest_entries(hidden_states_dir: str) -> list[dict]:
    """Collect entries from every manifest_*.json shard in the directory."""
    entries: list[dict] = []
    for mpath in sorted(glob.glob(os.path.join(hidden_states_dir, "manifest_*.json"))):
        with open(mpath) as f:
            man = json.load(f)
        entries.extend(man.get("entries", []))
    # Fallback: a single manifest.json.
    single = os.path.join(hidden_states_dir, "manifest.json")
    if not entries and os.path.exists(single):
        with open(single) as f:
            entries.extend(json.load(f).get("entries", []))
    return entries


def _build_batches(
    lengths: list[int],
    indices: list[int],
    max_samples: int,
    max_tokens: int,
) -> list[list[int]]:
    """Group equal-length samples into unpadded batches."""
    by_len: dict[int, list[int]] = {}
    for idx, ln in zip(indices, lengths):
        by_len.setdefault(ln, []).append(idx)
    batches: list[list[int]] = []
    for ln, idxs in by_len.items():
        cap = max(1, min(max_samples, max_tokens // max(1, ln)))
        for start in range(0, len(idxs), cap):
            batches.append(idxs[start : start + cap])
    return batches


class DFlashBatchDataset(Dataset):
    """Map-style dataset whose items are already-collated equal-length batches."""

    def __init__(
        self,
        cfg: DFlashConfig,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.cap = cfg.train_max_seq_len
        self.seed = seed
        self.epoch = 0

        entries = load_manifest_entries(cfg.hidden_states_dir)
        if not entries:
            raise RuntimeError(
                f"No manifest entries found under {cfg.hidden_states_dir}. "
                "Run extract.py first."
            )
        # Shard entries across data-parallel ranks.
        entries = entries[rank::world_size]
        self.entries = entries
        # Bucket length after cropping is min(num_tokens, cap).
        lengths = [min(int(e["num_tokens"]), self.cap) for e in entries]
        self.lengths = lengths
        self._all_indices = list(range(len(entries)))
        self.batches = _build_batches(
            lengths,
            self._all_indices,
            cfg.max_samples_per_batch,
            cfg.max_batch_tokens,
        )
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        self.batches = _build_batches(
            self.lengths,
            self._all_indices,
            self.cfg.max_samples_per_batch,
            self.cfg.max_batch_tokens,
        )
        rng.shuffle(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def _load_one(
        self, idx: int, target_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        entry = self.entries[idx]
        data = load_file(entry["path"], device="cpu")
        hidden = data["hidden_states"]  # [T, num_aux, hidden]
        token_ids = data["token_ids"].long()  # [T]
        target_last = data.get("target_last_hidden")  # [T, hidden] or None
        t = token_ids.shape[0]
        if t > target_len:
            start = random.randint(0, t - target_len)
            hidden = hidden[start : start + target_len]
            token_ids = token_ids[start : start + target_len]
            if target_last is not None:
                target_last = target_last[start : start + target_len]
        return token_ids, hidden, target_last

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        idxs = self.batches[batch_idx]
        target_len = min(self.lengths[idxs[0]], self.cap)
        # Recompute an exact common length (all bucket members share length).
        toks = []
        hids = []
        lasts = []
        for i in idxs:
            token_ids, hidden, target_last = self._load_one(i, target_len)
            # Members share a bucket length; guard against off-by-one shards.
            ln = token_ids.shape[0]
            if ln != target_len:
                ln = min(ln, target_len)
                token_ids = token_ids[:ln]
                hidden = hidden[:ln]
                if target_last is not None:
                    target_last = target_last[:ln]
                target_len = ln
            toks.append(token_ids)
            hids.append(hidden)
            lasts.append(target_last)
        # Enforce identical length across the batch after any trimming.
        common = min(t.shape[0] for t in toks)
        toks = [t[:common] for t in toks]
        hids = [h[:common] for h in hids]
        input_ids = torch.stack(toks, dim=0)          # [B, L]
        aux_hidden = torch.stack(hids, dim=0)          # [B, L, num_aux, hidden]
        result = {"input_ids": input_ids, "aux_hidden": aux_hidden}
        # Stack target_last_hidden if available (for L1 loss).
        if all(l is not None for l in lasts):
            lasts = [l[:common] for l in lasts]
            result["target_last_hidden"] = torch.stack(lasts, dim=0)  # [B, L, hidden]
        return result


def identity_collate(batch):
    # DataLoader is configured with batch_size=None; each item is a full batch.
    return batch
