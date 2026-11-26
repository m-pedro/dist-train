import copy
import math
from typing import Dict, List, Sequence
import torch
import torch.nn as nn

from dist_train.collectives import InMemoryGroup


class ZeROShardedTrainer:
    """Implements DeepSpeed ZeRO Stage 1 (Optimizer Sharding) and Stage 2 (+ Gradient Sharding)."""

    def __init__(
        self,
        base_model: nn.Module,
        world_size: int = 4,
        stage: int = 2,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if stage not in (1, 2):
            raise ValueError(f"ZeROShardedTrainer supports stage 1 or 2, got {stage}")
        self.world_size = world_size
        self.stage = stage
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.comm = InMemoryGroup(world_size=world_size)

        self.replicas: List[nn.Module] = [copy.deepcopy(base_model) for _ in range(world_size)]
        self.total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        self.padded_total = math.ceil(self.total_params / float(world_size)) * world_size
        self.shard_size = self.padded_total // world_size

        # Each rank allocates AdamW (m, v) ONLY for its 1/N local shard!
        self.step_count = 0
        self.exp_avg: List[torch.Tensor] = [
            torch.zeros(self.shard_size, dtype=torch.float32) for _ in range(world_size)
        ]
        self.exp_avg_sq: List[torch.Tensor] = [
            torch.zeros(self.shard_size, dtype=torch.float32) for _ in range(world_size)
        ]
        self.last_retained_grad_elems_per_rank: int = self.total_params

    def _flatten_params(self, model: nn.Module) -> torch.Tensor:
        flat = torch.cat([p.data.reshape(-1) for p in model.parameters() if p.requires_grad])
        pad = self.padded_total - flat.numel()
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
        return flat

    def _flatten_grads(self, model: nn.Module) -> torch.Tensor:
        pieces = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            g = p.grad if p.grad is not None else torch.zeros_like(p.data)
            pieces.append(g.reshape(-1))
        flat = torch.cat(pieces)
        pad = self.padded_total - flat.numel()
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
        return flat

    def _write_flat_params(self, model: nn.Module, flat_full: torch.Tensor) -> None:
        cursor = 0
        valid = flat_full[: self.total_params]
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.data.copy_(valid[cursor : cursor + n].view_as(p))
            cursor += n

    def per_rank_state_bytes(self) -> Dict[str, int]:
        """Returns steady-state per-rank memory breakdown (bytes) for weights, grads, and AdamW."""
        param_bytes = self.total_params * 4
        grad_bytes = (
            self.shard_size * 4 if self.stage == 2 else self.total_params * 4
        )
        optim_bytes = self.shard_size * 8  # m + v in fp32
        return {
            "param_bytes": param_bytes,
            "grad_bytes": grad_bytes,
            "optim_bytes": optim_bytes,
            "total_bytes": param_bytes + grad_bytes + optim_bytes,
        }

    def step_microbatches(
        self,
        inputs_per_rank: Sequence[torch.Tensor],
        targets_per_rank: Sequence[torch.Tensor],
        loss_fn,
    ) -> float:
        losses = []
        for r in range(self.world_size):
            for p in self.replicas[r].parameters():
                p.grad = None
            out = self.replicas[r](inputs_per_rank[r])
            loss = loss_fn(out, targets_per_rank[r])
            loss.backward()
            losses.append(float(loss.item()))

        flat_grads = [self._flatten_grads(m) for m in self.replicas]

        if self.stage == 1:
            # Stage 1: Full AllReduce on grads, then slice rank's 1/N shard for optimizer step
            reduced_full = self.comm.all_reduce(flat_grads, op="avg")
            grad_shards = [
                reduced_full[r][r * self.shard_size : (r + 1) * self.shard_size]
                for r in range(self.world_size)
            ]
            self.last_retained_grad_elems_per_rank = self.total_params
        else:
            # Stage 2: ReduceScatter grads directly into 1/N shards and immediately free full grads!
            grad_shards = self.comm.reduce_scatter(flat_grads, op="avg")
            for m in self.replicas:
                for p in m.parameters():
                    p.grad = None
            self.last_retained_grad_elems_per_rank = self.shard_size

        # Sharded AdamW update on each rank's owned 1/N slice
        self.step_count += 1
        b1, b2 = self.betas
        bc1 = 1.0 - (b1**self.step_count)
        bc2 = 1.0 - (b2**self.step_count)

        updated_param_shards: List[torch.Tensor] = []
        for r in range(self.world_size):
            p_full = self._flatten_params(self.replicas[r])
            p_shard = p_full[r * self.shard_size : (r + 1) * self.shard_size].clone()
            g_shard = grad_shards[r]

            # Decoupled AdamW weight decay + moment updates
            p_shard.mul_(1.0 - self.lr * self.weight_decay)
            m = self.exp_avg[r]
            v = self.exp_avg_sq[r]
            m.mul_(b1).add_(g_shard, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g_shard, g_shard, value=1.0 - b2)

            m_hat = m / bc1
            v_hat = v / bc2
            p_shard.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)
            updated_param_shards.append(p_shard)

        # Ring AllGather updated parameter shards back to all ranks
        gathered_params = self.comm.all_gather(updated_param_shards)
        for r in range(self.world_size):
            self._write_flat_params(self.replicas[r], gathered_params[r])

        return sum(losses) / float(len(losses))
