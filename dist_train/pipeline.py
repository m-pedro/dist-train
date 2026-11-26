from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn


@dataclass
class P2PChannelStats:
    """Tracks Point-to-Point activation (forward) and gradient (backward) transfers across stage links."""

    p2p_messages: int = 0
    p2p_bytes: int = 0

    def record_send(self, tensor: torch.Tensor) -> torch.Tensor:
        self.p2p_messages += 1
        self.p2p_bytes += tensor.numel() * tensor.element_size()
        # Detach across rank boundary (mirrors dist.send / dist.recv P2P wire transfer)
        return tensor.detach().clone()


class PipelineStage:
    """Represents one Pipeline-Parallel rank owning a contiguous slice of model layers."""

    def __init__(self, stage_id: int, num_stages: int, submodule: nn.Module) -> None:
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.module = submodule

        # Keyed by microbatch_id -> (detached_input, forward_output) awaiting backward()
        self._inflight_buffers: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.peak_inflight_microbatches: int = 0

    @property
    def is_first(self) -> bool:
        return self.stage_id == 0

    @property
    def is_last(self) -> bool:
        return self.stage_id == self.num_stages - 1

    @property
    def current_inflight(self) -> int:
        return len(self._inflight_buffers)

    def zero_grad(self) -> None:
        for p in self.module.parameters():
            p.grad = None

    def scale_grads(self, scale: float) -> None:
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    def forward_microbatch(self, mb_id: int, x: torch.Tensor) -> torch.Tensor:
        inp = x.detach().requires_grad_(not self.is_first)
        out = self.module(inp)
        self._inflight_buffers[mb_id] = (inp, out)
        self.peak_inflight_microbatches = max(
            self.peak_inflight_microbatches, len(self._inflight_buffers)
        )
        return out

    def backward_microbatch(
        self,
        mb_id: int,
        upstream_grad_or_loss: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Runs backward for `mb_id`, frees its cached forward activations, and returns dL/dx_in."""
        inp, out = self._inflight_buffers.pop(mb_id)
        if self.is_last and upstream_grad_or_loss.ndim == 0:
            # Scalar loss on final stage
            upstream_grad_or_loss.backward()
        else:
            out.backward(upstream_grad_or_loss)
        return inp.grad.detach().clone() if (not self.is_first and inp.grad is not None) else None


def partition_layers_into_stages(
    layers: Sequence[nn.Module], pp_size: int
) -> List[PipelineStage]:
    if pp_size <= 0 or len(layers) < pp_size:
        raise ValueError(
            f"len(layers) ({len(layers)}) must be >= pp_size ({pp_size}) and pp_size >= 1"
        )
    stages: List[PipelineStage] = []
    base = len(layers) // pp_size
    rem = len(layers) % pp_size
    cursor = 0
    for s in range(pp_size):
        cnt = base + (1 if s < rem else 0)
        sub = nn.Sequential(*layers[cursor : cursor + cnt])
        stages.append(PipelineStage(stage_id=s, num_stages=pp_size, submodule=sub))
        cursor += cnt
    return stages
