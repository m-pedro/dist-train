# dist-train

A Megatron-LM / DeepSpeed / PyTorch FSDP-style 3D Parallel (`DP` x `TP` x `PP`) and ZeRO Distributed LLM Training Engine built from scratch in Python and PyTorch. Implements Ring Collectives (`AllReduce`, `ReduceScatter`, `AllGather`), Bucketed `DDP`, ZeRO Stages 1, 2 & 3 (`FSDP`), Megatron `ColumnParallelLinear` / `RowParallelLinear` Tensor Parallelism, Activation Checkpointing, and 1F1B Interleaved Pipeline Parallelism.

## 15-Day Incremental Engineering Roadmap (Month 6, Days 151–165)

- [x] **Day 151:** Initialize `dist_train` package, 3D `DeviceMesh3D` topology, MIT license, and 15-day roadmap
- [ ] **Day 152:** Implement Ring `AllReduce`, `ReduceScatter`, `AllGather`, and `torch.distributed` `gloo` launcher (`dist_train/collectives.py`)
- [ ] **Day 153:** Implement `BucketedDDP` with reverse-order autograd hooks and flat gradient fusion (`dist_train/ddp.py`)
- [ ] **Day 154:** Implement `ZeRO-1` (sharded `AdamW` states) and `ZeRO-2` (`ReduceScatter` gradient sharding) (`dist_train/zero.py`)
- [ ] **Day 155:** Implement PyTorch `FSDP`-style (`ZeRO-3`) parameter sharding with `AllGather`/`ReduceScatter` hooks (`dist_train/fsdp.py`)
- [ ] **Day 156:** Implement Megatron-LM `ColumnParallelLinear` and `RowParallelLinear` (`dist_train/tensor_parallel.py`)
- [ ] **Day 157:** Implement Tensor-Parallel Multi-Head Attention and `SwiGLU` Transformer block
- [ ] **Day 158:** Implement selective activation checkpointing (`dist_train/checkpointing.py`)
- [ ] **Day 159:** Implement `PipelineStage` partitioner and P2P boundary messaging (`dist_train/pipeline.py`)
- [ ] **Day 160:** Implement `GPipe` and `1F1B` (One-Forward-One-Backward) micro-batch schedules
- [ ] **Day 161:** Compose 3D Parallelism (`DP/ZeRO` x `TP` x `PP`) unified training engine (`dist_train/engine_3d.py`)
- [ ] **Day 162:** Implement cluster VRAM & bandwidth profiler (`DDP` vs `ZeRO-1/2/3` vs `3D`)
- [ ] **Day 163:** Add multi-worker parity suite proving bit-exact step equivalence vs single-device `AdamW`
- [ ] **Day 164:** Build `dist-train` CLI (`bench-memory`, `demo-fsdp-gloo`, `demo-3d`)
- [ ] **Day 165:** Complete 3D Parallelism & ZeRO/FSDP architecture guide, collective math table, and benchmarks
