import copy
from typing import List, Sequence
import torch
import torch.nn as nn

from dist_train.collectives import InMemoryGroup


class BucketedDDPTrainer:
    """Data-Parallel coordinator fusing gradients into reverse-order flat buckets for Ring AllReduce."""

    def __init__(
        self,
        base_model: nn.Module,
        world_size: int = 4,
        bucket_cap_elems: int = 256,
        lr: float = 1e-2,
        weight_decay: float = 0.01,
    ) -> None:
        self.world_size = world_size
        self.bucket_cap_elems = bucket_cap_elems
        self.comm = InMemoryGroup(world_size=world_size)

        # Broadcast rank-0 initial weights identically across all replicas
        self.replicas: List[nn.Module] = [copy.deepcopy(base_model) for _ in range(world_size)]
        self.optimizers = [
            torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
            for m in self.replicas
        ]
        self.buckets: List[List[int]] = self._build_reverse_buckets(self.replicas[0])
        self.num_allreduce_calls: int = 0

    def _build_reverse_buckets(self, model: nn.Module) -> List[List[int]]:
        params = [p for p in model.parameters() if p.requires_grad]
        buckets: List[List[int]] = []
        cur_bucket: List[int] = []
        cur_elems = 0

        # Traverse in reverse order to match autograd backward execution order
        for p_idx in reversed(range(len(params))):
            n_el = params[p_idx].numel()
            if cur_bucket and (cur_elems + n_el > self.bucket_cap_elems):
                buckets.append(cur_bucket)
                cur_bucket = []
                cur_elems = 0
            cur_bucket.append(p_idx)
            cur_elems += n_el

        if cur_bucket:
            buckets.append(cur_bucket)
        return buckets

    def synchronize_gradients(self) -> None:
        """Flattens each parameter bucket, pads to world_size multiple, and Ring-AllReduces."""
        replica_params = [
            [p for p in m.parameters() if p.requires_grad] for m in self.replicas
        ]

        for bucket_indices in self.buckets:
            flat_per_rank: List[torch.Tensor] = []
            orig_len = 0

            for r in range(self.world_size):
                grads = []
                for idx in bucket_indices:
                    g = replica_params[r][idx].grad
                    if g is None:
                        g = torch.zeros_like(replica_params[r][idx])
                    grads.append(g.reshape(-1))
                flat = torch.cat(grads, dim=0)
                orig_len = flat.numel()
                rem = orig_len % self.world_size
                if rem != 0:
                    pad = self.world_size - rem
                    flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
                flat_per_rank.append(flat)

            reduced = self.comm.all_reduce(flat_per_rank, op="avg")
            self.num_allreduce_calls += 1

            # Unpack averaged flat bucket back into each replica's param.grad
            for r in range(self.world_size):
                cursor = 0
                buf = reduced[r][:orig_len]
                for idx in bucket_indices:
                    p = replica_params[r][idx]
                    cnt = p.numel()
                    p.grad = buf[cursor : cursor + cnt].view_as(p).clone()
                    cursor += cnt

    def step_microbatches(
        self,
        inputs_per_rank: Sequence[torch.Tensor],
        targets_per_rank: Sequence[torch.Tensor],
        loss_fn,
    ) -> float:
        losses = []
        for r in range(self.world_size):
            self.optimizers[r].zero_grad(set_to_none=True)
            out = self.replicas[r](inputs_per_rank[r])
            loss = loss_fn(out, targets_per_rank[r])
            loss.backward()
            losses.append(float(loss.item()))

        self.synchronize_gradients()
        for opt in self.optimizers:
            opt.step()
        return sum(losses) / float(len(losses))
