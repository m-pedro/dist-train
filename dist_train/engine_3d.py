from dataclasses import dataclass
from typing import Callable, List, Sequence
import torch
import torch.nn as nn

from dist_train.collectives import InMemoryGroup
from dist_train.mesh import DeviceMesh3D
from dist_train.pipeline import PipelineStage
from dist_train.pipeline_schedule import run_1f1b_step
from dist_train.tp_transformer import (
    ReferenceTransformerBlock,
    TensorParallelTransformerBlock,
)


class _ForkToTPBoundary(torch.autograd.Function):
    """Replicates single-tensor stage input across TP ranks in forward; averages identical rank dL/dx in backward."""

    @staticmethod
    def forward(ctx, tp_size: int, x: torch.Tensor):
        ctx.tp_size = tp_size
        return tuple(x.clone() for _ in range(tp_size))

    @staticmethod
    def backward(ctx, *grad_xs: torch.Tensor):
        avg_grad = torch.stack(list(grad_xs), dim=0).mean(dim=0)
        return None, avg_grad


class _JoinFromTPBoundary(torch.autograd.Function):
    """Emits unified stage output in forward; broadcasts full 1.0x upstream gradient to all TP ranks in backward."""

    @staticmethod
    def forward(ctx, *xs: torch.Tensor) -> torch.Tensor:
        ctx.tp_size = len(xs)
        return xs[0].clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return tuple(grad_output.clone() for _ in range(ctx.tp_size))


class _TPStageAdapter(nn.Module):
    """Adapts a sequence of TensorParallelTransformerBlocks to PipelineStage's single-tensor boundary API."""

    def __init__(self, tp_blocks: Sequence[TensorParallelTransformerBlock], tp_size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(list(tp_blocks))
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs: List[torch.Tensor] = list(_ForkToTPBoundary.apply(self.tp_size, x))
        for blk in self.blocks:
            xs = blk(xs)
        return _JoinFromTPBoundary.apply(*xs)


@dataclass(frozen=True)
class Step3DMetrics:
    loss: float
    world_size: int
    dp_bytes: int
    tp_bytes: int
    pp_p2p_bytes: int
    peak_inflight_stage0: int


class Parallel3DTrainer:
    """Unified 3D Parallel (DP x PP x TP) Transformer trainer executing 1F1B + Megatron TP + DP sync."""

    def __init__(
        self,
        master_blocks: Sequence[ReferenceTransformerBlock],
        mesh: DeviceMesh3D,
        num_microbatches: int = 4,
        lr: float = 1e-2,
        weight_decay: float = 0.01,
    ) -> None:
        if len(master_blocks) < mesh.pp_size or len(master_blocks) % mesh.pp_size != 0:
            raise ValueError(
                f"num_layers ({len(master_blocks)}) must be divisible by pp_size ({mesh.pp_size})"
            )
        self.mesh = mesh
        self.num_layers = len(master_blocks)
        self.layers_per_stage = self.num_layers // mesh.pp_size
        self.num_microbatches = num_microbatches

        self.dp_comm = InMemoryGroup(world_size=mesh.dp_size)
        # Dedicated TP communicator per (dp_rank, pp_rank) group
        self.tp_comms: List[List[InMemoryGroup]] = [
            [InMemoryGroup(world_size=mesh.tp_size) for _ in range(mesh.pp_size)]
            for _ in range(mesh.dp_size)
        ]

        # Build DP replicas of PP stage chains containing TP-sharded Transformer blocks
        self.dp_pipelines: List[List[PipelineStage]] = []
        self.optimizers: List[torch.optim.AdamW] = []

        for dp_r in range(mesh.dp_size):
            stages: List[PipelineStage] = []
            all_dp_params: List[nn.Parameter] = []
            for pp_r in range(mesh.pp_size):
                start = pp_r * self.layers_per_stage
                end = start + self.layers_per_stage
                tp_comm = self.tp_comms[dp_r][pp_r]
                tp_blks = [
                    TensorParallelTransformerBlock(
                        dim=blk.dim,
                        n_heads=blk.n_heads,
                        hidden_dim=blk.hidden_dim,
                        tp_size=mesh.tp_size,
                        comm=tp_comm,
                        master_block=blk,
                    )
                    for blk in master_blocks[start:end]
                ]
                adapter = _TPStageAdapter(tp_blks, tp_size=mesh.tp_size)
                stage = PipelineStage(stage_id=pp_r, num_stages=mesh.pp_size, submodule=adapter)
                stages.append(stage)
                all_dp_params.extend([p for p in adapter.parameters() if p.requires_grad])

            self.dp_pipelines.append(stages)
            self.optimizers.append(
                torch.optim.AdamW(all_dp_params, lr=lr, weight_decay=weight_decay)
            )

    def _sync_dp_gradients(self) -> None:
        if self.mesh.dp_size <= 1:
            return
        # Synchronize matching (pp_stage, tp_shard_param) across DP replicas!
        for pp_r in range(self.mesh.pp_size):
            stage_param_lists = [
                [p for p in self.dp_pipelines[dp_r][pp_r].module.parameters() if p.requires_grad]
                for dp_r in range(self.mesh.dp_size)
            ]
            num_params = len(stage_param_lists[0])
            for p_idx in range(num_params):
                grads = []
                orig_shape = stage_param_lists[0][p_idx].shape
                orig_numel = stage_param_lists[0][p_idx].numel()
                rem = orig_numel % self.mesh.dp_size
                pad = (self.mesh.dp_size - rem) if rem != 0 else 0

                for dp_r in range(self.mesh.dp_size):
                    g = stage_param_lists[dp_r][p_idx].grad
                    flat = g.reshape(-1) if g is not None else torch.zeros(orig_numel)
                    if pad > 0:
                        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
                    grads.append(flat)

                avg_grads = self.dp_comm.all_reduce(grads, op="avg")
                for dp_r in range(self.mesh.dp_size):
                    stage_param_lists[dp_r][p_idx].grad = (
                        avg_grads[dp_r][:orig_numel].view(orig_shape).clone()
                    )

    def step_batch(
        self,
        global_inputs: torch.Tensor,
        global_targets: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Step3DMetrics:
        dp_inputs = global_inputs.chunk(self.mesh.dp_size, dim=0)
        dp_targets = global_targets.chunk(self.mesh.dp_size, dim=0)

        dp_losses: List[float] = []
        p2p_bytes_total = 0
        peak_stage0 = 0

        for dp_r in range(self.mesh.dp_size):
            self.optimizers[dp_r].zero_grad(set_to_none=True)
            mb_x = dp_inputs[dp_r].chunk(self.num_microbatches, dim=0)
            mb_y = dp_targets[dp_r].chunk(self.num_microbatches, dim=0)

            rep = run_1f1b_step(self.dp_pipelines[dp_r], mb_x, mb_y, loss_fn)
            dp_losses.append(rep.mean_loss)
            p2p_bytes_total += rep.p2p_bytes
            peak_stage0 = max(peak_stage0, rep.peak_inflight_per_stage[0])

        self._sync_dp_gradients()
        for opt in self.optimizers:
            opt.step()

        tp_bytes = sum(
            self.tp_comms[d][p].bytes_transferred
            for d in range(self.mesh.dp_size)
            for p in range(self.mesh.pp_size)
        )
        return Step3DMetrics(
            loss=sum(dp_losses) / float(len(dp_losses)),
            world_size=self.mesh.world_size,
            dp_bytes=self.dp_comm.bytes_transferred,
            tp_bytes=tp_bytes,
            pp_p2p_bytes=p2p_bytes_total,
            peak_inflight_stage0=peak_stage0,
        )

    def export_dense_blocks(self) -> List[ReferenceTransformerBlock]:
        """Reconstructs unsharded ReferenceTransformerBlocks from DP replica 0's PP+TP shards."""
        reconstructed: List[ReferenceTransformerBlock] = []
        for pp_r in range(self.mesh.pp_size):
            adapter: _TPStageAdapter = self.dp_pipelines[0][pp_r].module  # type: ignore[assignment]
            for tp_blk in adapter.blocks:
                assert isinstance(tp_blk, TensorParallelTransformerBlock)
                dense = ReferenceTransformerBlock(
                    dim=tp_blk.dim,
                    n_heads=tp_blk.n_heads,
                    hidden_dim=tp_blk.w_gate.out_features,
                )
                with torch.no_grad():
                    dense.attn_norm_w.copy_(tp_blk.attn_norm_ws[0])
                    dense.wq.weight.copy_(torch.cat([w.detach() for w in tp_blk.wq.weights], dim=0))
                    dense.wk.weight.copy_(torch.cat([w.detach() for w in tp_blk.wk.weights], dim=0))
                    dense.wv.weight.copy_(torch.cat([w.detach() for w in tp_blk.wv.weights], dim=0))
                    dense.wo.weight.copy_(torch.cat([w.detach() for w in tp_blk.wo.weights], dim=1))

                    dense.ffn_norm_w.copy_(tp_blk.ffn_norm_ws[0])
                    dense.w_gate.weight.copy_(
                        torch.cat([w.detach() for w in tp_blk.w_gate.weights], dim=0)
                    )
                    dense.w_up.weight.copy_(
                        torch.cat([w.detach() for w in tp_blk.w_up.weights], dim=0)
                    )
                    dense.w_down.weight.copy_(
                        torch.cat([w.detach() for w in tp_blk.w_down.weights], dim=1)
                    )
                reconstructed.append(dense)
        return reconstructed
