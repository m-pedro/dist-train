import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_train.ddp import BucketedDDPTrainer
from dist_train.fsdp import FSDPTrainer


def test_fsdp_zero3_matches_single_device_adamw_and_slashes_memory_by_4x():
    torch.manual_seed(77)
    layers = [
        nn.Sequential(nn.Linear(32, 64), nn.GELU()),
        nn.Sequential(nn.Linear(64, 64), nn.GELU()),
        nn.Sequential(nn.Linear(64, 32), nn.GELU()),
        nn.Linear(32, 10),
    ]
    ref_model = nn.Sequential(*copy.deepcopy(layers))
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, weight_decay=0.01)

    total_model_elems = sum(p.numel() for p in ref_model.parameters())
    max_single_layer_elems = max(sum(p.numel() for p in layer.parameters()) for layer in layers)

    fsdp = FSDPTrainer(layers, world_size=4, lr=1e-2, weight_decay=0.01)
    ddp = BucketedDDPTrainer(ref_model, world_size=4, lr=1e-2, weight_decay=0.01)

    for _ in range(3):
        x = torch.randn(16, 32)
        y = torch.randint(0, 10, (16,))

        ref_opt.zero_grad()
        F.cross_entropy(ref_model(x), y).backward()
        ref_opt.step()

        fsdp.step_microbatches(x.chunk(4), y.chunk(4), F.cross_entropy)
        ddp.step_microbatches(x.chunk(4), y.chunk(4), F.cross_entropy)

    # 1. Verify exact numerical parity between 4-rank FSDP (ZeRO-3) and Single-GPU AdamW!
    exported_fsdp = fsdp.reconstruct_full_model()
    for p_ref, p_fsdp in zip(ref_model.parameters(), exported_fsdp.parameters()):
        assert torch.allclose(p_ref, p_fsdp, atol=1e-5)

    # 2. Verify sliding-window peak materialized weights equal ONLY the largest single layer, never the whole model!
    assert fsdp.peak_materialized_param_elems_per_rank == max_single_layer_elems
    assert fsdp.peak_materialized_param_elems_per_rank < total_model_elems // 2

    # 3. Verify steady-state per-rank memory is ~4x smaller than unsharded DDP (16 bytes/param -> 4 bytes/param)
    full_ddp_bytes = total_model_elems * 16
    fsdp_bytes = fsdp.steady_state_bytes_per_rank()["total_bytes"]
    assert fsdp_bytes <= (full_ddp_bytes // 4) + 256
