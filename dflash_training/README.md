# DFlash Draft Model Training Pipeline

Train a DFlash parallel-drafting draft model for speculative decoding with vLLM.

Target model: **Qwen3.6-27B-FP8** (64 layers, hidden=5120, vocab=248320).

## Overview

```
原始数据 ──▶ 数据生成 ──▶ 特征抽取 ──▶ 训练 ──▶ 导出 ──▶ 评测/性能
  (HF)     (sglang)    (vLLM KV)   (DDP)   (safetensors) (vLLM spec decode)
```

| Stage | Script | 输入 | 输出 |
|-------|--------|------|------|
| 1. 下载数据 | `dflash_training/scripts/data/download_and_split.py` | HuggingFace dataset | train/eval JSONL |
| 2. 生成回答 | `dflash_training/scripts/data/generate_train_data.py` | prompts JSONL | regen JSONL (target model 回答) |
| 3. 抽取特征 | `dflash_training/scripts/extract.sh` | regen JSONL | hidden state shards |
| 4. 训练+导出 | `dflash_training/scripts/train.sh` | hidden state shards | DFlashDraftModel checkpoint |
| 5a. 精度评测 | `dflash_training/scripts/eval.sh` | checkpoint | acceptance rate, 正确性 |
| 5b. 性能测试 | `dflash_training/scripts/test-dflash-qwen3-27b.sh` | checkpoint | throughput, TTFT, speedup |

---

## Prerequisites

```bash
# Python 环境 (需要 vllm, transformers, safetensors, sglang)
conda activate deepspec

# Target model
ls /root/Qwen3.6-27B-FP8

# GPU: 8x 32GB+ (5090/A100/H100)
nvidia-smi --query-gpu=name,memory.total --format=csv
```

---

## Stage 1: 下载和处理训练数据

### 1a. 下载并切分数据集

从 HuggingFace 下载 `mlabonne/open-perfectblend` 并按 95/5 切分 train/eval：

```bash
cd /root/vllm-dspark

python dflash_training/scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing
```

### 1b. 用 target model 重新生成 assistant 回答

先启动 sglang 推理服务器（8 workers, 每 worker 1 GPU）：

```bash
bash dflash_training/scripts/data/launch_sglang_server_qwen3_27b.sh
# 或 vLLM:
# bash dflash_training/scripts/data/launch_vllm_server_qwen3_27b.sh
```

等服务器就绪后，批量生成：

```bash
python dflash_training/scripts/data/generate_train_data.py \
    --model /root/Qwen3.6-27B-FP8 \
    --server-address 127.0.0.1:30000 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 \
                     127.0.0.1:30004 127.0.0.1:30005 127.0.0.1:30006 127.0.0.1:30007 \
    --concurrency 32 \
    --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3_27b/perfectblend_train_regen.jsonl
```

取子集（如 40k）：

```bash
head -40000 train_datasets/qwen3_27b/perfectblend_train_regen.jsonl \
    > train_datasets/qwen3_27b/perfectblend_train_regen_40k.jsonl
```

> **注意**：dflash_training 的 extract.py 直接读 regen JSONL，不需要 deepspec 框架的
> target cache（`prepare_target_cache.py`）。Step 3 是给 deepspec 训练框架用的。

### 一键脚本

以上两步也可以用 pipeline 脚本串联（需要先停 sglang 再跑 Step 3，dflash 不需要 Step 3）：

```bash
bash dflash_training/scripts/data/prepare_data_qwen3_27b.sh
```

---

## Stage 2: 抽取 Target Hidden States

用 vLLM 的 KV transfer 机制，让 target model 跑一遍推理，提取中间层 hidden states。

```bash
export TARGET_MODEL_PATH=/root/Qwen3.6-27B-FP8
export TRAIN_DATA_PATH=train_datasets/qwen3_27b/perfectblend_train_regen_40k.jsonl
export HIDDEN_STATES_DIR=/mnt/deepspec/qwen3_27b_dflash_hidden
export AUX_LAYER_IDS=8,20,32,44,56          # 要提取的 target 层
export TENSOR_PARALLEL_SIZE=2                # 每个 worker 的 TP 大小
export NUM_SHARDS=4                           # 数据并行 worker 数 (4 x TP2 = 8 GPUs)

bash dflash_training/scripts/extract.sh
```

**输出**：`${HIDDEN_STATES_DIR}/` 下每个 request 一个 `.safetensors` 文件 + `manifest.json`。

### 可选：提取 last hidden state（用于 L1 loss）

```bash
export EXTRACT_TARGET_LAST_HIDDEN=1
# 需要重新跑 extract（会在每个 shard 里增加 target_last_hidden 字段）
bash dflash_training/scripts/extract.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TARGET_MODEL_PATH` | `/root/Qwen3.6-27B-FP8` | Target model 路径 |
| `TRAIN_DATA_PATH` | (见 config.py) | Regen JSONL 路径 |
| `HIDDEN_STATES_DIR` | `/mnt/deepspec/qwen3_27b_dflash_hidden` | 输出目录 |
| `AUX_LAYER_IDS` | `8,20,32,44,56` | 提取的 target 层 ID |
| `MAX_SEQ_LEN` | `4096` | 最大序列长度 |
| `TENSOR_PARALLEL_SIZE` | `2` | TP 大小 |
| `GPU_MEMORY_UTIL` | `0.90` | GPU 显存利用率 |
| `NUM_SHARDS` | `4` | 数据并行 worker 数 |
| `EXTRACT_TARGET_LAST_HIDDEN` | `0` | 是否提取最后一层 hidden state |

---

## Stage 3: 训练

DDP 8-GPU 训练 DFlash draft model，训练结束后自动导出 checkpoint。

```bash
export TARGET_MODEL_PATH=/root/Qwen3.6-27B-FP8
export HIDDEN_STATES_DIR=/mnt/deepspec/qwen3_27b_dflash_hidden
export OUTPUT_DIR=/mnt/deepspec/qwen3_27b_dflash_ckpt

# Draft 结构
export BLOCK_SIZE=8                    # 每个 anchor 预测的 future token 数
export NUM_DRAFT_LAYERS=1              # Draft decoder 层数
export MASK_TOKEN_ID=248319            # Parallel-drafting mask token

# Loss 设计
export CE_LOSS_ALPHA=1.0               # CE loss 权重
export L1_LOSS_ALPHA=0.0               # L1 loss 权重 (需要 EXTRACT_TARGET_LAST_HIDDEN=1)
export LOSS_DECAY_GAMMA=0.0            # 位置衰减 γ (0=禁用, paper=4.0)

# Markov head (半自回归, 0=禁用, 256=paper default)
export MARKOV_RANK=0

# 优化
export LR=1e-4
export MAX_STEPS=20000
export WARMUP_STEPS=200

bash dflash_training/scripts/train.sh
```

### 输出

`${OUTPUT_DIR}/` 目录：
```
config.json           — architectures=["DFlashDraftModel"] + dflash_config
model.safetensors     — draft backbone 权重 (fc, hidden_norm, norm, layers.*)
mask_embedding.pt     — 训练后的 mask embedding
markov_head.pt        — Markov head 权重 (仅当 MARKOV_RANK > 0)
```

### 训练超参

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BLOCK_SIZE` | `8` | Draft block size (每 anchor 预测 token 数) |
| `NUM_DRAFT_LAYERS` | `1` | Draft decoder 层数 |
| `MASK_TOKEN_ID` | `248319` | Reserved mask token ID |
| `MARKOV_RANK` | `0` | Markov head rank (0=纯 DFlash) |
| `CE_LOSS_ALPHA` | `1.0` | CE loss 权重 |
| `L1_LOSS_ALPHA` | `0.0` | L1 loss 权重 (需 last hidden) |
| `LOSS_DECAY_GAMMA` | `0.0` | 位置衰减 (paper=4.0) |
| `LR` | `1e-4` | 学习率 |
| `MAX_STEPS` | `20000` | 最大训练步数 |
| `TRAIN_MAX_SEQ_LEN` | `256` | 训练序列长度 |
| `MAX_BATCH_TOKENS` | `1024` | 每 batch 最大 token 数 |

---

## Stage 4: 导出

导出在 `train.sh` 训练循环内自动完成（见 `export.py`）。每个 checkpoint interval
和训练结束都会导出 vLLM 原生格式的 `DFlashDraftModel` checkpoint。

验证导出是否正确：

```bash
python -m dflash_training.validate_checkpoint \
    --checkpoint-dir /mnt/deepspec/qwen3_27b_dflash_ckpt
```

---

## Stage 5: 测试推理

### 5a. 精度评测 — Acceptance Rate

评测 draft model 在 speculative decoding 下的 acceptance rate 和输出正确性：

```bash
export TARGET_MODEL=/root/Qwen3.6-27B-FP8
export DRAFT_MODEL=/mnt/deepspec/qwen3_27b_dflash_ckpt
export TENSOR_PARALLEL_SIZE=2
export NUM_SPEC_TOKENS=8

bash dflash_training/scripts/eval.sh
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TARGET_MODEL` | `/root/Qwen3.6-27B-FP8` | Target model |
| `DRAFT_MODEL` | `/mnt/deepspec/qwen3_27b_dflash_ckpt` | Draft checkpoint |
| `TENSOR_PARALLEL_SIZE` | `2` | TP 大小 |
| `NUM_SPEC_TOKENS` | `8` | Speculative tokens 数 |
| `NUM_SAMPLES` | `100` | 评测样本数 |
| `MAX_NEW_TOKENS` | `512` | 最大生成 token 数 |

### 5b. 性能 Benchmark — Throughput & Speedup

对比 AR baseline vs DFlash speculative decoding 的吞吐、TTFT、TPOP：

```bash
export TARGET_MODEL=/root/Qwen3.6-27B-FP8
export DRAFT_MODEL=/mnt/deepspec/qwen3_27b_dflash_ckpt
export TENSOR_PARALLEL_SIZE=2
export NUM_PROMPTS=32

bash dflash_training/scripts/test-dflash-qwen3-27b.sh
```

测试两组配置：
- `input=4096, output=2048`（长输入短输出）
- `input=512, output=4096`（短输入长输出）

输出 AR 和 DFlash 的 TTFT、TPOP、decode throughput 及 speedup 倍数。

---

## Quick Start (End-to-End)

```bash
cd /root/vllm-dspark

# 1. 数据准备 (已有数据可跳过)
python dflash_training/scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets --skip-existing

# 2. 生成 target 回答 (需先启动 sglang 服务器)
bash dflash_training/scripts/data/launch_sglang_server_qwen3_27b.sh
python dflash_training/scripts/data/generate_train_data.py \
    --model /root/Qwen3.6-27B-FP8 \
    --server-address 127.0.0.1:{30000..30007} \
    --concurrency 32 --temperature 0.7 --top-p 0.8 --max-tokens 4096 \
    --disable-thinking --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3_27b/perfectblend_train_regen.jsonl
head -40000 train_datasets/qwen3_27b/perfectblend_train_regen.jsonl \
    > train_datasets/qwen3_27b/perfectblend_train_regen_40k.jsonl

# 3. 抽取 hidden states
export TRAIN_DATA_PATH=train_datasets/qwen3_27b/perfectblend_train_regen_40k.jsonl
bash dflash_training/scripts/extract.sh

# 4. 训练
bash dflash_training/scripts/train.sh

# 5. 评测
bash dflash_training/scripts/eval.sh
bash dflash_training/scripts/test-dflash-qwen3-27b.sh
```
