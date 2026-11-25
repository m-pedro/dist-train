import torch
import torch.distributed as dist
from dist_train.collectives import (
    InMemoryGroup,
    ring_all_gather,
    ring_all_reduce,
    ring_reduce_scatter,
    spawn_gloo_workers,
)


def _real_gloo_allreduce_worker(rank: int, world_size: int, base_vec: list[float]) -> list[float]:
    t = torch.tensor(base_vec, dtype=torch.float32) * float(rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def test_ring_collectives_and_real_gloo_multiprocess_parity():
    t0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ranks = [t0 * float(r + 1) for r in range(4)]  # Multipliers 1, 2, 3, 4 -> sum = 10 * t0

    # 1. Ring ReduceScatter + Ring AllGather == Ring AllReduce == exact sum
    shards = ring_reduce_scatter(ranks, op="sum")
    assert len(shards) == 4 and all(s.numel() == 2 for s in shards)
    assert torch.allclose(shards[0], torch.tensor([10.0, 20.0]))
    assert torch.allclose(shards[3], torch.tensor([70.0, 80.0]))

    gathered = ring_all_gather(shards)
    for g in gathered:
        assert torch.allclose(g, t0 * 10.0)

    comm = InMemoryGroup(world_size=4)
    avg_out = comm.all_reduce(ranks, op="avg")
    assert torch.allclose(avg_out[0], t0 * 2.5)
    assert comm.bytes_transferred > 0

    # 2. Spawn 2 real OS processes with torch.distributed gloo and verify collective agreement
    gloo_results = spawn_gloo_workers(2, _real_gloo_allreduce_worker, [1.0, 2.0, 3.0, 4.0])
    # Rank 0 contributed 1x, Rank 1 contributed 2x -> sum = 3x on both OS processes!
    assert gloo_results[0] == [3.0, 6.0, 9.0, 12.0]
    assert gloo_results[1] == [3.0, 6.0, 9.0, 12.0]
