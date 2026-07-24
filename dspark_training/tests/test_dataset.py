# SPDX-License-Identifier: Apache-2.0
import json
import os

import torch
from safetensors.torch import save_file

from dspark_training.config import DSparkConfig
from dspark_training.dataset import DSparkBatchDataset


def test_batches_have_no_target_last_hidden(tmp_path):
    cfg = DSparkConfig()
    cfg.hidden_states_dir = str(tmp_path)
    cfg.max_samples_per_batch = 2
    cfg.max_batch_tokens = 1000
    cfg.train_max_seq_len = 16
    for i in range(2):
        save_file(
            {
                "hidden_states": torch.randn(
                    16, cfg.num_aux_layers, cfg.hidden_size
                ),
                "token_ids": torch.randint(0, 100, (16,)),
            },
            os.path.join(tmp_path, f"shard_000_{i:07d}.safetensors"),
        )
    json.dump(
        {
            "entries": [
                {
                    "id": i,
                    "path": os.path.join(tmp_path, f"shard_000_{i:07d}.safetensors"),
                    "num_tokens": 16,
                }
                for i in range(2)
            ]
        },
        open(os.path.join(tmp_path, "manifest_000.json"), "w"),
    )
    ds = DSparkBatchDataset(cfg, rank=0, world_size=1, seed=0)
    batch = ds[0]
    assert set(batch.keys()) == {"input_ids", "aux_hidden"}


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_batches_have_no_target_last_hidden(Path(d))
    print("PASS test_batches_have_no_target_last_hidden")
