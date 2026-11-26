from typing import List, Sequence
import torch
import torch.nn as nn


class _CheckpointSegmentFn(torch.autograd.Function):
    """Custom autograd Function discarding intermediate activations during forward and recomputing in backward."""

    @staticmethod
    def forward(ctx, run_module: nn.Module, x: torch.Tensor, *module_params: torch.Tensor) -> torch.Tensor:
        ctx.run_module = run_module
        # Save ONLY the single boundary input tensor `x` (zero internal layer activations saved!)
        ctx.save_for_backward(x)
        with torch.no_grad():
            return run_module(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (saved_x,) = ctx.saved_tensors
        detached_x = saved_x.detach().requires_grad_(saved_x.requires_grad)

        with torch.enable_grad():
            recomputed_out = ctx.run_module(detached_x)

        params = [p for p in ctx.run_module.parameters() if p.requires_grad]
        inputs_to_diff = ([detached_x] if detached_x.requires_grad else []) + params
        grads = torch.autograd.grad(
            outputs=recomputed_out,
            inputs=inputs_to_diff,
            grad_outputs=grad_output,
            allow_unused=True,
        )

        if detached_x.requires_grad:
            x_grad = grads[0]
            param_grads = grads[1:]
        else:
            x_grad = None
            param_grads = grads

        return (None, x_grad, *param_grads)


def checkpoint_segment(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Executes `module(x)` with activation checkpointing."""
    params = [p for p in module.parameters() if p.requires_grad]
    return _CheckpointSegmentFn.apply(module, x, *params)


class CheckpointedSequential(nn.Module):
    """Partitions `layers` into segments of `checkpoint_every` layers to bound saved activations to O(L / k)."""

    def __init__(self, layers: Sequence[nn.Module], checkpoint_every: int = 2) -> None:
        super().__init__()
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be >= 1")
        self.segments = nn.ModuleList(
            [
                nn.Sequential(*layers[i : i + checkpoint_every])
                for i in range(0, len(layers), checkpoint_every)
            ]
        )
        self.saved_boundary_tensors_count: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure `x` requires grad so interior segment parameters receive gradients cleanly
        h = x if x.requires_grad else x.detach().requires_grad_(True)
        self.saved_boundary_tensors_count = len(self.segments)
        for seg in self.segments:
            h = checkpoint_segment(seg, h)
        return h
