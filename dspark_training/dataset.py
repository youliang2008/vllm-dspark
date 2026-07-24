# SPDX-License-Identifier: Apache-2.0
"""Read extracted hidden-state shards into DSpark training batches.

Each shard (written by extract.py via vLLM's ExampleHiddenStatesConnector) holds:
    hidden_states : [num_tokens, num_aux_layers, hidden_size]
    token_ids     : [num_tokens]

No-padding equal-length batches (length-bucketing). DSpark does not use
``target_last_hidden`` (no L1 loss), so it is never read.
"""
from __future__ import annotations

import glob
import json
import os
import random

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from .config import DSparkConfig


def load_manifest_entries(hidden_states_dir: str) -> list[dict]:
    entries: list[dict] = []
    for mpath in sorted(glob.glob(os.path.join(hidden_states_dir, "manifest_*.json"))):
        with open(mpath) as f:
            man = json.load(f)
        entries.extend(man.get("entries", []))
    single = os.path.join(hidden_states_dir, "manifest.json")
    if not entries and os.path.exists(single):
        with open(single) as f:
            entries.extend(json.load(f).get("entries", []))
    return entries


def _build_batches(
    lengths: list[int], indices: list[int], max_samples: int, max_tokens: int
) -> list[list[int]]:
    by_len: dict[int, list[int]] = {}
    for idx, ln in zip(indices, lengths):
        by_len.setdefault(ln, []).append(idx)
    batches: list[list[int]] = []
    for ln, idxs in by_len.items():
        cap = max(1, min(max_samples, max_tokens // max(1, ln)))
        for start in range(0, len(idxs), cap):
            batches.append(idxs[start : start + cap])
    return batches


class DSparkBatchDataset(Dataset):
    """Map-style dataset whose items are already-collated equal-length batches."""

    def __init__(
        self,
        cfg: DSparkConfig,
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
        entries = entries[rank::world_size]
        self.entries = entries
        lengths = [min(int(e["num_tokens"]), self.cap) for e in entries]
        self.lengths = lengths
        self._all_indices = list(range(len(entries)))
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        data = load_file(entry["path"], device="cpu")
        hidden = data["hidden_states"]
        token_ids = data["token_ids"].long()
        t = token_ids.shape[0]
        if t > target_len:
            start = random.randint(0, t - target_len)
            hidden = hidden[start : start + target_len]
            token_ids = token_ids[start : start + target_len]
        return token_ids, hidden

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        idxs = self.batches[batch_idx]
        target_len = min(self.lengths[idxs[0]], self.cap)
        toks = []
        hids = []
        for i in idxs:
            token_ids, hidden = self._load_one(i, target_len)
            ln = token_ids.shape[0]
            if ln != target_len:
                ln = min(ln, target_len)
                token_ids = token_ids[:ln]
                hidden = hidden[:ln]
                target_len = ln
            toks.append(token_ids)
            hids.append(hidden)
        common = min(t.shape[0] for t in toks)
        toks = [t[:common] for t in toks]
        hids = [h[:common] for h in hids]
        input_ids = torch.stack(toks, dim=0)
        aux_hidden = torch.stack(hids, dim=0)
        return {"input_ids": input_ids, "aux_hidden": aux_hidden}
