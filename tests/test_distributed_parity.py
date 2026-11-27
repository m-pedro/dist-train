import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from dist_train.collectives import spawn_gloo_workers
from dist_train.ddp import BucketedDDPTrainer
from dist_train.fsdp import FSDPTrainer
from dist_train.zero import ZeROShardedTrainer


def _gloo_zero2_multiprocess_worker(rank: int, world_size: int, payload: dict) -> list[float]:
    """Executes real multi-process ZeRO-2 using torch.distributed reduce_scatter + all_gather."""
    torch.manual_seed(payload["model_seed"])
    layers = [nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)]
    model = nn.Sequential(*layers)

    params = [p for p in model.parameters() if p.requires_grad]
    total_elems = sum(p.numel() for p in params)
    pad = (world_size - (total_elems % world_size)) % world_size
    padded_total = total_elems + pad
    shard_size = padded_total // world_size

    lr = payload["lr"]
    wd = payload["wd"]
    b1, b2 = 0.9, 0.999
    exp_avg = torch.zeros(shard_size, dtype=torch.float32)
    exp_avg_sq = torch.zeros(shard_size, dtype=torch.float32)

    for step_idx, (x_batch, y_batch) in enumerate(payload["batches"], start=1):
        x_local = x_batch.chunk(world_size, dim=0)[rank]
        y_local = y_batch.chunk(world_size, dim=0)[rank]

        for p in params:
            p.grad = None
        loss = F.cross_entropy(model(x_local), y_local)
        loss.backward()

        flat_grad = torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None])
        if pad > 0:
            flat_grad = torch.cat([flat_grad, torch.zeros(pad)])

        # 1. Real OS-process ReduceScatter across gloo TCP sockets!
        grad_shard = torch.zeros(shard_size, dtype=torch.float32)
        dist.reduce_scatter_tensor(grad_shard, flat_grad / float(world_size), op=dist.ReduceOp.SUM)

        # 2. Local 1/N Sharded AdamW update
        flat_param = torch.cat([p.data.reshape(-1) for p in params])
        if pad > 0:
            flat_param = torch.cat([flat_param, torch.zeros(pad)])
        p_shard = flat_param[rank * shard_size : (rank + 1) * shard_size].clone()

        p_shard.mul_(1.0 - lr * wd)
        exp_avg.mul_(b1).add_(grad_shard, alpha=1.0 - b1)
        exp_avg_sq.mul_(b2).addcmul_(grad_shard, grad_shard, value=1.0 - b2)
        m_hat = exp_avg / (1.0 - (b1**step_idx))
        v_hat = exp_avg_sq / (1.0 - (b2**step_idx))
        p_shard.addcdiv_(m_hat, v_hat.sqrt().add_(1e-8), value=-lr)

        # 3. Real OS-process AllGather of updated parameter shards!
        gathered = torch.zeros(padded_total, dtype=torch.float32)
        dist.all_gather_into_tensor(gathered, p_shard)

        cursor = 0
        for p in params:
            n = p.numel()
            p.data.copy_(gathered[cursor : cursor + n].view_as(p))
            cursor += n

    return torch.cat([p.detach().reshape(-1) for p in params]).tolist()


def test_unified_multi_step_parity_across_ddp_zero1_zero2_fsdp_and_real_gloo_processes():
    model_seed = 404
    torch.manual_seed(model_seed)
    layers = [nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)]
    ref_model = nn.Sequential(*copy.deepcopy(layers))
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, weight_decay=0.01)

    ddp = BucketedDDPTrainer(ref_model, world_size=2, lr=1e-2, weight_decay=0.01)
    z1 = ZeROShardedTrainer(ref_model, world_size=2, stage=1, lr=1e-2, weight_decay=0.01)
    z2 = ZeROShardedTrainer(ref_model, world_size=2, stage=2, lr=1e-2, weight_decay=0.01)
    fsdp = FSDPTrainer(copy.deepcopy(layers), world_size=2, lr=1e-2, weight_decay=0.01)

    batches = [
        (torch.randn(8, 16), torch.randint(0, 8, (8,)))
        for _ in range(3)
    ]

    for x, y in batches:
        ref_opt.zero_grad()
        F.cross_entropy(ref_model(x), y).backward()
        ref_opt.step()

        ddp.step_microbatches(x.chunk(2), y.chunk(2), F.cross_entropy)
        z1.step_microbatches(x.chunk(2), y.chunk(2), F.cross_entropy)
        z2.step_microbatches(x.chunk(2), y.chunk(2), F.cross_entropy)
        fsdp.step_microbatches(x.chunk(2), y.chunk(2), F.cross_entropy)

    # Spawn 2 real OS worker processes running Gloo ReduceScatter + AllGather!
    gloo_outs = spawn_gloo_workers(
        2,
        _gloo_zero2_multiprocess_worker,
        {"model_seed": model_seed, "lr": 1e-2, "wd": 0.01, "batches": batches},
    )

    ref_flat = torch.cat([p.detach().reshape(-1) for p in ref_model.parameters()])
    ddp_flat = torch.cat([p.detach().reshape(-1) for p in ddp.replicas[0].parameters()])
    z1_flat = torch.cat([p.detach().reshape(-1) for p in z1.replicas[0].parameters()])
    z2_flat = torch.cat([p.detach().reshape(-1) for p in z2.replicas[0].parameters()])
    fsdp_flat = torch.cat([p.detach().reshape(-1) for p in fsdp.reconstruct_full_model().parameters()])
    gloo_r0 = torch.tensor(gloo_outs[0])
    gloo_r1 = torch.tensor(gloo_outs[1])

    assert torch.allclose(ref_flat, ddp_flat, atol=1e-5)
    assert torch.allclose(ref_flat, z1_flat, atol=1e-5)
    assert torch.allclose(ref_flat, z2_flat, atol=1e-5)
    assert torch.allclose(ref_flat, fsdp_flat, atol=1e-5)
    assert torch.allclose(ref_flat, gloo_r0, atol=1e-5)
    assert torch.allclose(gloo_r0, gloo_r1, atol=1e-6)
