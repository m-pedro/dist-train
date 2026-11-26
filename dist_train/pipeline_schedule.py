from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple
import torch

from dist_train.pipeline import P2PChannelStats, PipelineStage


@dataclass(frozen=True)
class PipelineStepReport:
    mean_loss: float
    bubble_fraction: float
    peak_inflight_per_stage: List[int]
    p2p_messages: int
    p2p_bytes: int


def theoretical_pipeline_bubble_fraction(pp_size: int, num_microbatches: int) -> float:
    """Computes idle pipeline bubble fraction (P - 1) / (M + P - 1) for M micro-batches on P stages."""
    if pp_size <= 1:
        return 0.0
    return float(pp_size - 1) / float(num_microbatches + pp_size - 1)


def run_gpipe_step(
    stages: Sequence[PipelineStage],
    micro_inputs: Sequence[torch.Tensor],
    micro_targets: Sequence[torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> PipelineStepReport:
    """Executes a GPipe step (all M forwards across all stages, then all M backwards)."""
    num_mb = len(micro_inputs)
    pp_size = len(stages)
    p2p = P2PChannelStats()

    for st in stages:
        st.zero_grad()
        st.peak_inflight_microbatches = 0

    # 1. Forward ALL M micro-batches end-to-end (accumulates M in-flight activations per stage!)
    losses_t: List[torch.Tensor] = []
    for m in range(num_mb):
        act = micro_inputs[m]
        for s in range(pp_size):
            out = stages[s].forward_microbatch(m, act)
            if s < pp_size - 1:
                act = p2p.record_send(out)
            else:
                losses_t.append(loss_fn(out, micro_targets[m]))

    # 2. Backward ALL M micro-batches from last stage back to Stage 0
    for m in range(num_mb):
        grad = stages[-1].backward_microbatch(m, losses_t[m])
        for s in reversed(range(pp_size - 1)):
            assert grad is not None
            wire_grad = p2p.record_send(grad)
            grad = stages[s].backward_microbatch(m, wire_grad)

    # Average accumulated micro-batch gradients by 1 / M
    inv_m = 1.0 / float(num_mb)
    for st in stages:
        st.scale_grads(inv_m)

    return PipelineStepReport(
        mean_loss=sum(float(l.item()) for l in losses_t) * inv_m,
        bubble_fraction=theoretical_pipeline_bubble_fraction(pp_size, num_mb),
        peak_inflight_per_stage=[st.peak_inflight_microbatches for st in stages],
        p2p_messages=p2p.p2p_messages,
        p2p_bytes=p2p.p2p_bytes,
    )


def run_1f1b_step(
    stages: Sequence[PipelineStage],
    micro_inputs: Sequence[torch.Tensor],
    micro_targets: Sequence[torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> PipelineStepReport:
    """Executes Megatron-style clock-cycle 1F1B (Warmup -> Steady-State 1F1B -> Cooldown).

    Simulates discrete pipeline clock ticks across `P` stages so Stage `s` has at most
    `P - s` micro-batches in flight simultaneously (bounding peak activation memory to O(P) vs O(M)!).
    """
    num_mb = len(micro_inputs)
    pp_size = len(stages)
    p2p = P2PChannelStats()

    for st in stages:
        st.zero_grad()
        st.peak_inflight_microbatches = 0

    # Per-stage action queues built according to exact 1F1B schedule:
    # Stage s performs `warmup = min(P - s - 1, M)` F-steps, then `(M - warmup)` 1F+1B pairs, then `warmup` B-steps.
    schedules: List[List[Tuple[str, int]]] = []
    for s in range(pp_size):
        warmup = min(pp_size - s - 1, num_mb)
        steady = num_mb - warmup
        ops: List[Tuple[str, int]] = []
        f_idx = 0
        b_idx = 0
        for _ in range(warmup):
            ops.append(("F", f_idx))
            f_idx += 1
        for _ in range(steady):
            ops.append(("F", f_idx))
            f_idx += 1
            ops.append(("B", b_idx))
            b_idx += 1
        for _ in range(warmup):
            ops.append(("B", b_idx))
            b_idx += 1
        schedules.append(ops)

    # Wire mailboxes keyed by (dst_stage, mb_id)
    fwd_Activation_box: Dict[Tuple[int, int], torch.Tensor] = {
        (0, m): micro_inputs[m] for m in range(num_mb)
    }
    bwd_grad_box: Dict[Tuple[int, int], torch.Tensor] = {}
    stage_losses: Dict[int, torch.Tensor] = {}
    loss_vals: List[float] = []

    cursors = [0] * pp_size
    total_ops = 2 * num_mb

    # Advance pipeline event loop until all stages finish all 2*M operations
    while any(c < total_ops for c in cursors):
        progressed = False
        for s in range(pp_size):
            if cursors[s] >= total_ops:
                continue
            kind, m = schedules[s][cursors[s]]

            if kind == "F":
                if (s, m) not in fwd_Activation_box:
                    continue
                inp = fwd_Activation_box.pop((s, m))
                out = stages[s].forward_microbatch(m, inp)
                if s < pp_size - 1:
                    fwd_Activation_box[(s + 1, m)] = p2p.record_send(out)
                else:
                    l_t = loss_fn(out, micro_targets[m])
                    stage_losses[m] = l_t
                    loss_vals.append(float(l_t.item()))
                cursors[s] += 1
                progressed = True

            else:  # kind == "B"
                if s == pp_size - 1:
                    if m not in stage_losses:
                        continue
                    l_t = stage_losses.pop(m)
                    g_prev = stages[s].backward_microbatch(m, l_t)
                else:
                    if (s, m) not in bwd_grad_box:
                        continue
                    g_in = bwd_grad_box.pop((s, m))
                    g_prev = stages[s].backward_microbatch(m, g_in)

                if s > 0 and g_prev is not None:
                    bwd_grad_box[(s - 1, m)] = p2p.record_send(g_prev)
                cursors[s] += 1
                progressed = True

        if not progressed:
            raise RuntimeError("1F1B pipeline schedule deadlocked")

    inv_m = 1.0 / float(num_mb)
    for st in stages:
        st.scale_grads(inv_m)

    return PipelineStepReport(
        mean_loss=sum(loss_vals) * inv_m,
        bubble_fraction=theoretical_pipeline_bubble_fraction(pp_size, num_mb),
        peak_inflight_per_stage=[st.peak_inflight_microbatches for st in stages],
        p2p_messages=p2p.p2p_messages,
        p2p_bytes=p2p.p2p_bytes,
    )
