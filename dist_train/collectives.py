import os
import socket
from typing import Any, Callable, List, Sequence
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def ring_reduce_scatter(rank_tensors: Sequence[torch.Tensor], op: str = "sum") -> List[torch.Tensor]:
    """Executes an exact N-1 step Ring ReduceScatter across `N` rank tensors of identical shape."""
    n = len(rank_tensors)
    if n == 0:
        return []
    numel = rank_tensors[0].numel()
    if numel % n != 0:
        raise ValueError(f"tensor numel ({numel}) must be divisible by world_size ({n})")

    # Split each rank's flat buffer into N equal chunks
    buffers: List[List[torch.Tensor]] = [
        [c.clone() for c in r.flatten().chunk(n)] for r in rank_tensors
    ]

    # N - 1 ring accumulation steps: in step s, rank r sends chunk (r - s) % N to (r + 1) % N
    for step in range(n - 1):
        outgoing = [
            (r, (r - step) % n, buffers[r][(r - step) % n].clone()) for r in range(n)
        ]
        for r, chunk_idx, payload in outgoing:
            recv_rank = (r + 1) % n
            buffers[recv_rank][chunk_idx] = buffers[recv_rank][chunk_idx] + payload

    # After N - 1 steps, rank r holds the fully reduced shard at index (r + 1) % N.
    # Permute canonically so rank r receives shard index `r` (matching dist.reduce_scatter_tensor).
    # Specifically, chunk `c` ends up on rank `(c + n - 1) % n`; rotate by +1 hop or map canonically:
    canonical_shards: List[torch.Tensor] = [torch.empty_like(buffers[0][0]) for _ in range(n)]
    for r in range(n):
        finished_chunk_idx = (r - (n - 1)) % n
        canonical_shards[finished_chunk_idx] = buffers[r][finished_chunk_idx]

    if op == "avg":
        canonical_shards = [s / float(n) for s in canonical_shards]
    elif op != "sum":
        raise ValueError(f"unsupported op '{op}'")
    return canonical_shards


def ring_all_gather(rank_shards: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Executes an exact N-1 step Ring AllGather assembling full tensors on all N ranks."""
    n = len(rank_shards)
    if n == 0:
        return []

    slots: List[List[torch.Tensor | None]] = [[None] * n for _ in range(n)]
    for r in range(n):
        slots[r][r] = rank_shards[r].flatten().clone()

    for step in range(n - 1):
        outgoing = []
        for r in range(n):
            send_idx = (r - step) % n
            shard_tensor = slots[r][send_idx]
            assert shard_tensor is not None
            outgoing.append((r, send_idx, shard_tensor.clone()))
        for r, send_idx, payload in outgoing:
            recv_rank = (r + 1) % n
            slots[recv_rank][send_idx] = payload

    return [torch.cat([s for s in slots[r] if s is not None], dim=0) for r in range(n)]


def ring_all_reduce(rank_tensors: Sequence[torch.Tensor], op: str = "sum") -> List[torch.Tensor]:
    """Bandwidth-optimal Ring AllReduce implemented as Ring ReduceScatter + Ring AllGather."""
    orig_shape = rank_tensors[0].shape
    scattered = ring_reduce_scatter(rank_tensors, op=op)
    gathered = ring_all_gather(scattered)
    return [g.view(orig_shape) for g in gathered]


class InMemoryGroup:
    """Deterministic multi-rank collective communicator with exact byte-volume accounting."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.bytes_transferred: int = 0

    def all_reduce(self, tensors: Sequence[torch.Tensor], op: str = "sum") -> List[torch.Tensor]:
        n = self.world_size
        if n == 1:
            return [tensors[0].clone()]
        elem_bytes = tensors[0].numel() * tensors[0].element_size()
        # Each rank sends 2 * (N - 1) / N * Bytes across the ring
        self.bytes_transferred += int(2 * (n - 1) * elem_bytes)
        return ring_all_reduce(tensors, op=op)

    def reduce_scatter(self, tensors: Sequence[torch.Tensor], op: str = "sum") -> List[torch.Tensor]:
        n = self.world_size
        if n == 1:
            return [tensors[0].flatten().clone()]
        elem_bytes = tensors[0].numel() * tensors[0].element_size()
        self.bytes_transferred += int((n - 1) * elem_bytes)
        return ring_reduce_scatter(tensors, op=op)

    def all_gather(self, shards: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        n = self.world_size
        if n == 1:
            return [shards[0].flatten().clone()]
        full_bytes = shards[0].numel() * n * shards[0].element_size()
        self.bytes_transferred += int((n - 1) * full_bytes)
        return ring_all_gather(shards)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _gloo_worker_entry(
    rank: int,
    world_size: int,
    port: int,
    user_fn: Callable[[int, int, Any], Any],
    payload: Any,
    return_queue: Any,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        res = user_fn(rank, world_size, payload)
        return_queue.put((rank, res))
    finally:
        dist.destroy_process_group()


def spawn_gloo_workers(
    world_size: int,
    user_fn: Callable[[int, int, Any], Any],
    payload: Any = None,
) -> List[Any]:
    """Spawns `world_size` real OS processes communicating via `torch.distributed` (gloo)."""
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    mp.spawn(
        _gloo_worker_entry,
        args=(world_size, port, user_fn, payload, queue),
        nprocs=world_size,
        join=True,
    )
    results = [None] * world_size
    for _ in range(world_size):
        r, val = queue.get()
        results[r] = val
    return results
