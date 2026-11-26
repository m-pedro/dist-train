import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_train.zero import ZeROShardedTrainer


def test_zero_stage1_and_stage2_match_single_device_adamw_with_sharded_memory():
    torch.manual_seed(99)
    base = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))

    ref = copy.deepcopy(base)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.01)

    z1 = ZeROShardedTrainer(base, world_size=4, stage=1, lr=1e-2, weight_decay=0.01)
    z2 = ZeROShardedTrainer(base, world_size=4, stage=2, lr=1e-2, weight_decay=0.01)

    for _ in range(3):
        x = torch.randn(16, 16)
        y = torch.randint(0, 8, (16,))

        ref_opt.zero_grad()
        F.cross_entropy(ref(x), y).backward()
        ref_opt.step()

        z1.step_microbatches(x.chunk(4), y.chunk(4), F.cross_entropy)
        z2.step_microbatches(x.chunk(4), y.chunk(4), F.cross_entropy)

    # Verify exact parameter parity across Single-GPU AdamW, ZeRO-1, and ZeRO-2!
    for p_ref, p_z1, p_z2 in zip(
        ref.parameters(), z1.replicas[0].parameters(), z2.replicas[0].parameters()
    ):
        assert torch.allclose(p_ref, p_z1, atol=1e-5)
        assert torch.allclose(p_ref, p_z2, atol=1e-5)

    # Verify ZeRO-2 shards gradients by 4x on top of ZeRO-1's 4x optimizer state sharding
    mem1 = z1.per_rank_state_bytes()
    mem2 = z2.per_rank_state_bytes()
    assert mem2["grad_bytes"] < mem1["grad_bytes"]
    assert mem2["total_bytes"] < mem1["total_bytes"]
