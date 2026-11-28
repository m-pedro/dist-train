# dist-train

A Megatron-LM / DeepSpeed / PyTorch FSDP-style 3D Parallel (`DP` x `PP` x `TP`) and ZeRO Distributed LLM Training Engine built from scratch in Python and PyTorch. Implements Ring Collectives (`AllReduce`, `ReduceScatter`, `AllGather`), Bucketed `DDP`, ZeRO Stages 1, 2 & 3 (`FSDP`), Megatron `ColumnParallelLinear` / `RowParallelLinear` Tensor Parallelism, Selective Activation Checkpointing, 1F1B Interleaved Pipeline Parallelism, and real multi-process OS training via `torch.distributed` (`gloo`).

---

## Repository Structure

```text
dist-train/
├── LICENSE                         # MIT License
├── README.md                       # 3D Parallelism & ZeRO/FSDP architecture & benchmarks
├── pyproject.toml                  # Package metadata & pytest configuration
├── dist_train/
│   ├── __init__.py                 # Public module exports
│   ├── mesh.py                     # DeviceMesh3D orthogonal (DP, PP, TP) process groups
│   ├── collectives.py              # Ring AllReduce/ReduceScatter/AllGather & gloo spawner
│   ├── ddp.py                      # BucketedDDPTrainer with reverse-order gradient fusion
│   ├── zero.py                     # DeepSpeed ZeRO Stage 1 & Stage 2 sharded AdamW engine
│   ├── fsdp.py                     # PyTorch FSDP (ZeRO-3) sliding-window parameter sharding
│   ├── tensor_parallel.py          # Megatron ColumnParallelLinear & RowParallelLinear (f/g)
│   ├── tp_transformer.py           # Tensor-Parallel Causal MHA & SwiGLU Transformer block
│   ├── checkpointing.py            # Selective activation checkpointing autograd Function
│   ├── pipeline.py                 # PipelineStage layer partitioner & P2P wire accounting
│   ├── pipeline_schedule.py        # GPipe vs clock-cycle 1F1B micro-batch schedulers
│   ├── engine_3d.py                # Unified Parallel3DTrainer (DP x PP x TP + 1F1B)
│   ├── profiler.py                 # Analytical 7B/70B VRAM & collective bandwidth profiler
│   └── cli.py                      # CLI (bench-memory, demo-fsdp-gloo, demo-3d)
└── tests/
    ├── test_mesh.py                # 3D DeviceMesh rank unraveling & group orthogonality
    ├── test_collectives.py         # Ring math & real 2-process gloo AllReduce tests
    ├── test_ddp.py                 # Multi-bucket fused DDP vs single-device AdamW parity
    ├── test_zero.py                # ZeRO-1 & ZeRO-2 sharded state & parity verification
    ├── test_fsdp.py                # FSDP peak single-layer materialization & 4x RAM cut
    ├── test_tensor_parallel.py     # Column/Row parallel MLP forward & backward parity
    ├── test_tp_transformer.py      # Head-sharded Attention + SwiGLU gradient equivalence
    ├── test_checkpointing.py       # >60% saved-activation cut via PyTorch tensor hooks
    ├── test_pipeline_stage.py      # Multi-stage P2P boundary activation & gradient tests
    ├── test_pipeline_schedule.py   # 1F1B O(P) vs GPipe O(M) in-flight memory theorem
    ├── test_engine_3d.py           # 8-rank (2x2x2) 3D Parallelism exact step parity
    ├── test_profiler.py            # Monotonic ZeRO savings & 1.5x FSDP comm ratio tests
    └── test_distributed_parity.py  # Unified DDP/ZeRO-1/2/FSDP + real OS gloo parity suite
```

---

## 3D Parallelism & ZeRO / FSDP Cluster Architecture

```text
  ┌───────────────────────────────────────────────────────────────────┐
  │ 1. 3D DeviceMesh Topology (World = 8: DP = 2 x PP = 2 x TP = 2)   │
  │                                                                   │
  │   DP Replica 0:  Stage 0 [Rank 0 ◄═TP═► Rank 1] ──1F1B P2P──┐     │
  │                                                             ▼     │
  │                  Stage 1 [Rank 2 ◄═TP═► Rank 3] ◄──dL/dx────┘     │
  │                             ▲              ▲                      │
  │                     DP/ZeRO │              │ DP/ZeRO              │
  │                             ▼              ▼                      │
  │   DP Replica 1:  Stage 1 [Rank 6 ◄═TP═► Rank 7] ◄──1F1B P2P─┐     │
  │                  Stage 0 [Rank 4 ◄═TP═► Rank 5] ────────────┘     │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ 2. Progressive ZeRO / PyTorch FSDP Sharding (Across N Workers)    │
  │   • Bucketed DDP : 16 Psi bytes/GPU  (Ring AllReduce: 4*(N-1)/N)  │
  │   • ZeRO-1 / Z-2 : Shards AdamW (12 Psi/N) + Grads (2 Psi/N)      │
  │   • FSDP (ZeRO-3): Shards Params + Grads + Optim (16 Psi/N total) │
  └───────────────────────────────────────────────────────────────────┘
```

---

## Distributed Memory & Collective Bandwidth Complexity (`bf16` + `fp32` `AdamW`)

For a model of `Psi` parameters trained across `N` workers (`2 Psi` bytes `bf16` weights, `2 Psi` bytes `bf16` gradients, `12 Psi` bytes `fp32` master weights + `m` + `v`):

| Parallelism Mode | Steady-State Model Memory / GPU | Step Communication Volume / GPU | Foundational Reference |
| :--- | :--- | :--- | :--- |
| **`Bucketed DDP`** | `2 Psi + 2 Psi + 12 Psi = 16 Psi` | `2 * (N - 1)/N * (2 Psi)` (`Ring AllReduce`) | Li et al., *PyTorch Distributed* (VLDB 2020) |
| **`ZeRO Stage 1`** | `2 Psi + 2 Psi + (12 Psi / N)` | `2 * (N - 1)/N * (2 Psi)` (`RS + AG`) | Rajbhandari et al., *ZeRO* (SC 2020) |
| **`ZeRO Stage 2`** | `2 Psi + (2 Psi + 12 Psi) / N` | `2 * (N - 1)/N * (2 Psi)` (`RS + AG`) | Rajbhandari et al., *ZeRO* (SC 2020) |
| **`FSDP (ZeRO-3)`** | `(2 Psi + 2 Psi + 12 Psi) / N` | `3 * (N - 1)/N * (2 Psi)` (`2x AG + 1x RS`) | Zhao et al., *PyTorch FSDP* (VLDB 2023) |
| **`Megatron TP`** | `Weights & Acts split by 1 / TP` | `4 AllReduces / layer` over fast intra-node NVLink | Shoeybi et al., *Megatron-LM* (2019) |
| **`1F1B Pipeline`** | `Params / PP`, in-flight acts `<= PP` | Point-to-Point `Send/Recv`, bubble `(P-1)/(M+P-1)` | Narayanan et al., *Megatron 3D* (SC 2021) |

---

## 7B-Parameter LLM Memory & Bandwidth Benchmark (`8x80GB GPUs`)

Generated via `python3 -m dist_train.cli bench-memory` (`Psi = 7B`, `L = 32`, `D = 4096`, `T = 2048`, `micro_batch = 2`, activation checkpointing enabled):

| Strategy (`World = 8`) | Weights | Gradients | `AdamW` States | Activations | Total VRAM / GPU | Comm / Step | Fits `80 GB` H100? |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`DDP (DP=8)`** | 13.04 GB | 13.04 GB | 78.23 GB | 1.00 GB | 105.31 GB | 22.82 GB | OOM (> 80 GB) |
| **`ZeRO-1 (DP=8)`** | 13.04 GB | 13.04 GB | 9.78 GB | 1.00 GB | 36.86 GB | 22.82 GB | YES (2.86x cut) |
| **`ZeRO-2 (DP=8)`** | 13.04 GB | 1.63 GB | 9.78 GB | 1.00 GB | 25.45 GB | 22.82 GB | YES (4.14x cut) |
| **`FSDP / ZeRO-3 (DP=8)`** | 2.04 GB | 2.04 GB | 9.78 GB | 1.00 GB | 14.85 GB | 34.23 GB | YES (7.09x cut) |
| **`3D (DP=2, PP=2, TP=2)`** | 3.26 GB | 3.26 GB | 9.78 GB | 0.50 GB | 16.80 GB | 5.31 GB | YES (6.27x cut) |

---

## Quickstart

### 1. Run All 13 Unit, Multi-Process `gloo` & 3D Parity Suites

```bash
source .venv/bin/activate
pytest -v
```

### 2. Run Cluster Profiler, Real 4-Process `gloo` FSDP & 8-Rank 3D Demos

```bash
python3 -m dist_train.cli bench-memory
python3 -m dist_train.cli demo-fsdp-gloo
python3 -m dist_train.cli demo-3d
```

---

## 15-Day Incremental Engineering Roadmap (Month 6, Days 151–165)

- [x] **Day 151:** Initialize `dist_train` package, 3D `DeviceMesh3D` topology, MIT license, and 15-day roadmap
- [x] **Day 152:** Implement Ring `AllReduce`, `ReduceScatter`, `AllGather`, and `torch.distributed` `gloo` launcher (`dist_train/collectives.py`)
- [x] **Day 153:** Implement `BucketedDDPTrainer` with reverse-order gradient fusion (`dist_train/ddp.py`)
- [x] **Day 154:** Implement `ZeRO-1` (sharded `AdamW` states) and `ZeRO-2` (`ReduceScatter` gradient sharding) (`dist_train/zero.py`)
- [x] **Day 155:** Implement PyTorch `FSDP`-style (`ZeRO-3`) parameter sharding (`dist_train/fsdp.py`)
- [x] **Day 156:** Implement Megatron-LM `ColumnParallelLinear` and `RowParallelLinear` (`dist_train/tensor_parallel.py`)
- [x] **Day 157:** Implement Tensor-Parallel Multi-Head Attention and `SwiGLU` block (`dist_train/tp_transformer.py`)
- [x] **Day 158:** Implement selective activation checkpointing (`dist_train/checkpointing.py`)
- [x] **Day 159:** Implement `PipelineStage` partitioner and P2P boundary messaging (`dist_train/pipeline.py`)
- [x] **Day 160:** Implement `GPipe` and `1F1B` micro-batch pipeline schedules (`dist_train/pipeline_schedule.py`)
- [x] **Day 161:** Compose 3D Parallelism (`DP` x `PP` x `TP`) unified training engine (`dist_train/engine_3d.py`)
- [x] **Day 162:** Implement cluster VRAM & bandwidth profiler (`dist_train/profiler.py`)
- [x] **Day 163:** Add multi-worker parity suite with real OS `gloo` process spawning (`tests/test_distributed_parity.py`)
- [x] **Day 164:** Build `dist-train` CLI (`bench-memory`, `demo-fsdp-gloo`, `demo-3d`) (`dist_train/cli.py`)
- [x] **Day 165:** Complete 3D Parallelism & ZeRO/FSDP architecture guide, collective math table, and benchmarks
