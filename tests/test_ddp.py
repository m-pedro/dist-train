import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_train.ddp import BucketedDDPTrainer


def test_bucketed_ddp_matches_single_device_full_batch_adamw_exactly():
    torch.manual_seed(42)
    base = nn.Sequential(
        nn.Linear(16, 32),
        nn.GELU(),
        nn.Linear(32, 16),
        nn.GELU(),
        nn.Linear(16, 8),
    )
    single_model = copy.deepcopy(base)
    single_opt = torch.optim.AdamW(single_model.parameters(), lr=1e-2, weight_decay=0.01)

    # Small bucket cap (128 elems) forces multiple fused gradient buckets
    ddp = BucketedDDPTrainer(base, world_size=4, bucket_cap_elems=128, lr=1e-2, weight_decay=0.01)
    assert len(ddp.buckets) >= 2

    x_full = torch.randn(16, 16)
    y_full = torch.randint(0, 8, (16,))

    # Single-device step on full batch of 16
    single_opt.zero_grad()
    loss_ref = F.cross_entropy(single_model(x_full), y_full)
    loss_ref.backward()
    single_opt.step()

    # 4-way BucketedDDP step on 4 micro-shards of 4 examples each
    loss_ddp = ddp.step_microbatches(
        x_full.chunk(4, dim=0),
        y_full.chunk(4, dim=0),
        F.cross_entropy,
    )

    assert abs(loss_ddp - float(loss_ref.item())) < 1e-5
    for p_ref, p_rep0, p_rep3 in zip(
        single_model.parameters(), ddp.replicas[0].parameters(), ddp.replicas[3].parameters()
    ):
        assert torch.allclose(p_ref, p_rep0, atol=1e-5)
        assert torch.allclose(p_rep0, p_rep3, atol=1e-6)
