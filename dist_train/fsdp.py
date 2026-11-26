import copy
import math
from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn

from dist_train.collectives import InMemoryGroup


class FSDPShardedUnit:
    """Manages per-rank 1/N parameter, gradient, and AdamW shards for one layer across `world_size`."""

    def __init__(self, replica_modules: List[nn.Module], comm: InMemoryGroup) -> None:
        self.modules = replica_modules
        self.comm = comm
        self.world_size = comm.world_size

        # Record parameter shapes and flatten initial weights
        self.param_meta: List[Tuple[str, torch.Size, int]] = []
        flat_pieces = []
        for name, p in self.modules[0].named_parameters():
            if p.requires_grad:
                self.param_meta.append((name, p.shape, p.numel()))
                flat_pieces.append(p.detach().reshape(-1))

        full_init = torch.cat(flat_pieces) if flat_pieces else torch.zeros(0)
        self.total_elems = full_init.numel()
        self.padded_elems = math.ceil(self.total_elems / float(self.world_size)) * self.world_size
        self.shard_size = self.padded_elems // self.world_size

        if self.padded_elems > self.total_elems:
            full_init = torch.cat(
                [full_init, torch.zeros(self.padded_elems - self.total_elems, dtype=full_init.dtype)]
            )

        # Store ONLY the 1/N local shard per rank and immediately release full module weights!
        self.param_shards: List[torch.Tensor] = [
            full_init[r * self.shard_size : (r + 1) * self.shard_size].clone()
            for r in range(self.world_size)
        ]
        self.grad_shards: List[torch.Tensor] = [
            torch.zeros(self.shard_size, dtype=torch.float32) for _ in range(self.world_size)
        ]
        self.exp_avg: List[torch.Tensor] = [
            torch.zeros(self.shard_size, dtype=torch.float32) for _ in range(self.world_size)
        ]
        self.exp_avg_sq: List[torch.Tensor] = [
            torch.zeros(self.shard_size, dtype=torch.float32) for _ in range(self.world_size)
        ]
        self.free_unsharded_weights()

    def materialize_weights(self) -> None:
        """AllGathers 1/N parameter shards into full temporary layer weights before compute."""
        if self.total_elems == 0:
            return
        gathered = self.comm.all_gather(self.param_shards)
        for r, mod in enumerate(self.modules):
            buf = gathered[r][: self.total_elems]
            cursor = 0
            param_dict = dict(mod.named_parameters())
            for name, shape, cnt in self.param_meta:
                param_dict[name].data = buf[cursor : cursor + cnt].view(shape).clone()
                cursor += cnt

    def free_unsharded_weights(self) -> None:
        """Releases full layer parameter tensors back to 0-element storage after compute."""
        for mod in self.modules:
            param_dict = dict(mod.named_parameters())
            for name, _, _ in self.param_meta:
                param_dict[name].data = torch.empty(0, dtype=torch.float32)

    def reduce_scatter_and_free_grads(self) -> None:
        """ReduceScatters full layer gradients into 1/N shards and frees unsharded grads + weights."""
        if self.total_elems == 0:
            return
        flat_grads: List[torch.Tensor] = []
        for mod in self.modules:
            param_dict = dict(mod.named_parameters())
            pieces = []
            for name, _, cnt in self.param_meta:
                g = param_dict[name].grad
                pieces.append(g.reshape(-1) if g is not None else torch.zeros(cnt))
                param_dict[name].grad = None
            flat = torch.cat(pieces)
            if self.padded_elems > self.total_elems:
                flat = torch.cat([flat, torch.zeros(self.padded_elems - self.total_elems)])
            flat_grads.append(flat)

        self.grad_shards = self.comm.reduce_scatter(flat_grads, op="avg")
        self.free_unsharded_weights()

    def step_adamw(
        self, step_num: int, lr: float, betas: Tuple[float, float], eps: float, weight_decay: float
    ) -> None:
        b1, b2 = betas
        bc1 = 1.0 - (b1**step_num)
        bc2 = 1.0 - (b2**step_num)

        for r in range(self.world_size):
            p = self.param_shards[r]
            g = self.grad_shards[r]
            p.mul_(1.0 - lr * weight_decay)
            self.exp_avg[r].mul_(b1).add_(g, alpha=1.0 - b1)
            self.exp_avg_sq[r].mul_(b2).addcmul_(g, g, value=1.0 - b2)
            m_hat = self.exp_avg[r] / bc1
            v_hat = self.exp_avg_sq[r] / bc2
            p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


class FSDPTrainer:
    """PyTorch FSDP / ZeRO-3 layer-by-layer sharded training engine."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        world_size: int = 4,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.world_size = world_size
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.comm = InMemoryGroup(world_size=world_size)

        self.units: List[FSDPShardedUnit] = []
        for layer in layers:
            replicas = [copy.deepcopy(layer) for _ in range(world_size)]
            self.units.append(FSDPShardedUnit(replicas, self.comm))

        self.step_count = 0
        self.peak_materialized_param_elems_per_rank: int = 0

    def steady_state_bytes_per_rank(self) -> Dict[str, int]:
        sharded_elems = sum(u.shard_size for u in self.units)
        param_b = sharded_elems * 4
        grad_b = sharded_elems * 4
        optim_b = sharded_elems * 8
        return {
            "param_bytes": param_b,
            "grad_bytes": grad_b,
            "optim_bytes": optim_b,
            "total_bytes": param_b + grad_b + optim_b,
        }

    def _current_unsharded_elems_rank0(self) -> int:
        total = 0
        for u in self.units:
            for p in u.modules[0].parameters():
                total += p.data.numel()
        return total

    def step_microbatches(
        self,
        inputs_per_rank: Sequence[torch.Tensor],
        targets_per_rank: Sequence[torch.Tensor],
        loss_fn,
    ) -> float:
        # 1. Layer-by-layer FSDP Forward: AllGather unit_l -> forward -> free unit_l weights!
        activations: List[List[torch.Tensor]] = [[x.detach()] for x in inputs_per_rank]

        with torch.no_grad():
            for unit in self.units:
                unit.materialize_weights()
                curr_live = self._current_unsharded_elems_rank0()
                self.peak_materialized_param_elems_per_rank = max(
                    self.peak_materialized_param_elems_per_rank, curr_live
                )

                for r in range(self.world_size):
                    out = unit.modules[r](activations[r][-1])
                    activations[r].append(out.detach())

                # Immediately discard unsharded weights so only 1 layer is ever materialized at once!
                unit.free_unsharded_weights()

        # Verify 0 unsharded parameters remain resident between forward and backward
        assert self._current_unsharded_elems_rank0() == 0

        # 2. Compute loss on detached leaf output to obtain upstream gradient dL/dh_L cleanly
        losses = []
        upstream_grads: List[torch.Tensor] = []
        for r in range(self.world_size):
            leaf_out = activations[r][-1].detach().requires_grad_(True)
            loss = loss_fn(leaf_out, targets_per_rank[r])
            loss.backward()
            losses.append(float(loss.item()))
            assert leaf_out.grad is not None
            upstream_grads.append(leaf_out.grad.detach())

        # 3. Reverse Layer-by-Layer FSDP Backward: AllGather unit_l -> recompute local graph -> backward -> ReduceScatter -> free!
        for l_idx in reversed(range(len(self.units))):
            unit = self.units[l_idx]
            unit.materialize_weights()
            next_upstream: List[torch.Tensor] = []

            for r in range(self.world_size):
                saved_inp = activations[r][l_idx].detach().requires_grad_(l_idx > 0)
                re_out = unit.modules[r](saved_inp)
                re_out.backward(upstream_grads[r])
                if l_idx > 0:
                    assert saved_inp.grad is not None
                    next_upstream.append(saved_inp.grad.detach())

            unit.reduce_scatter_and_free_grads()
            upstream_grads = next_upstream

        # 4. Sharded AdamW step on each unit's 1/N shards
        self.step_count += 1
        for unit in self.units:
            unit.step_adamw(
                self.step_count, self.lr, self.betas, self.eps, self.weight_decay
            )

        return sum(losses) / float(len(losses))

    def reconstruct_full_model(self) -> nn.Sequential:
        """Materializes and exports a clean single-device nn.Sequential copy for evaluation."""
        exported = []
        for unit in self.units:
            unit.materialize_weights()
            exported.append(copy.deepcopy(unit.modules[0]))
            unit.free_unsharded_weights()
        return nn.Sequential(*exported)
