# SPDX-License-Identifier: Apache-2.0
"""Self-contained DFlash training pipeline that produces checkpoints natively
loadable by vLLM's ``DFlashDraftModel`` (architectures=["DFlashDraftModel"]).

Pipeline stages:
    1. extract.py  - run vLLM's native ``extract_hidden_states`` over prompts to
                     dump per-request target hidden states (.safetensors).
    2. dataset.py  - read the extracted shards into training samples.
    3. dflash_draft_model.py - plain-PyTorch draft whose state_dict keys map 1:1
                     to vLLM's checkpoint contract.
    4. train.py    - FSDP training loop + block cross-entropy loss.
    5. export.py   - write config.json + safetensors in DFlashDraftModel format.
"""
