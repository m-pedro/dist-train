from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ClusterMemoryProfile:
    strategy: str
    world_size: int
    weight_gb: float
    grad_gb: float
    optim_gb: float
    activation_gb: float
    comm_gb_per_step: float

    @property
    def total_vram_gb(self) -> float:
        return self.weight_gb + self.grad_gb + self.optim_gb + self.activation_gb

    def fits_80gb_h100(self) -> bool:
        return self.total_vram_gb <= 80.0


def estimate_strategy_profile(
    strategy: str,
    num_params: float = 7e9,
    n_layers: int = 32,
    hidden_dim: int = 4096,
    seq_len: int = 2048,
    micro_batch_size: int = 2,
    dp_size: int = 8,
    tp_size: int = 1,
    pp_size: int = 1,
    activation_checkpointing: bool = True,
) -> ClusterMemoryProfile:
    """Computes mixed-precision (bf16 compute + fp32 AdamW) per-GPU GB and step communication GB."""
    gb = float(1024**3)
    world = dp_size * tp_size * pp_size

    # Base mixed-precision state sizes across the full model:
    # bf16 weights = 2*Psi, bf16 grads = 2*Psi, fp32 master + m + v = 12*Psi
    full_w = 2.0 * num_params
    full_g = 2.0 * num_params
    full_opt = 12.0 * num_params

    # Approximate per-layer Transformer activation footprint (bf16):
    # Without checkpointing ~ 16 * B * T * D per layer; with checkpointing ~ 2 * B * T * D boundary only!
    act_factor = 2.0 if activation_checkpointing else 16.0
    base_act_bytes = n_layers * act_factor * micro_batch_size * seq_len * hidden_dim

    if strategy == "ddp":
        n = world
        w_b, g_b, o_b = full_w, full_g, full_opt
        act_b = base_act_bytes
        # Ring AllReduce on bf16 grads: 2 * (N - 1) / N * (2 * Psi)
        comm_b = 2.0 * ((n - 1) / float(max(1, n))) * full_g

    elif strategy == "zero1":
        n = world
        w_b, g_b, o_b = full_w, full_g, full_opt / float(n)
        act_b = base_act_bytes
        # ReduceScatter(grads) + AllGather(bf16 updated weights) = 2 * (N - 1) / N * (2 * Psi)
        comm_b = 2.0 * ((n - 1) / float(max(1, n))) * full_g

    elif strategy == "zero2":
        n = world
        w_b, g_b, o_b = full_w, full_g / float(n), full_opt / float(n)
        act_b = base_act_bytes
        comm_b = 2.0 * ((n - 1) / float(max(1, n))) * full_g

    elif strategy == "fsdp":
        n = world
        # Steady-state 1/N shards + 1 single-layer unsharded weight + grad buffer
        single_layer_buf = (full_w + full_g) / float(max(1, n_layers))
        w_b = (full_w / float(n)) + (single_layer_buf * 0.5)
        g_b = (full_g / float(n)) + (single_layer_buf * 0.5)
        o_b = full_opt / float(n)
        act_b = base_act_bytes
        # Forward AllGather + Backward AllGather + Backward ReduceScatter = 3 * (N - 1) / N * (2 * Psi)
        comm_b = 3.0 * ((n - 1) / float(max(1, n))) * full_w

    elif strategy == "3d":
        model_shards = float(tp_size * pp_size)
        w_b = full_w / model_shards
        g_b = full_g / model_shards
        o_b = full_opt / float(world)  # ZeRO-1 across DP x TP x PP!
        # Stage 0 holds `pp_size` 1F1B microbatches over `n_layers / pp_size` layers, split by `tp_size`
        act_b = (base_act_bytes / float(pp_size)) * float(pp_size) / float(tp_size)

        dp_comm = (
            2.0 * ((dp_size - 1) / float(max(1, dp_size))) * g_b if dp_size > 1 else 0.0
        )
        # 4 TP AllReduces per layer (2 fwd + 2 bwd) of bf16 (B * T * D)
        tp_comm = (
            (n_layers / float(pp_size))
            * 4.0
            * 2.0
            * ((tp_size - 1) / float(max(1, tp_size)))
            * (2.0 * micro_batch_size * seq_len * hidden_dim)
            if tp_size > 1
            else 0.0
        )
        pp_comm = (
            2.0 * (2.0 * micro_batch_size * seq_len * hidden_dim) if pp_size > 1 else 0.0
        )
        comm_b = dp_comm + tp_comm + pp_comm

    else:
        raise ValueError(f"unknown strategy '{strategy}'")

    return ClusterMemoryProfile(
        strategy=strategy,
        world_size=world,
        weight_gb=w_b / gb,
        grad_gb=g_b / gb,
        optim_gb=o_b / gb,
        activation_gb=act_b / gb,
        comm_gb_per_step=comm_b / gb,
    )


def compare_7b_cluster_strategies() -> List[ClusterMemoryProfile]:
    """Compares an 8-GPU cluster training a 7B-parameter Transformer across all 5 strategies."""
    return [
        estimate_strategy_profile("ddp", dp_size=8, tp_size=1, pp_size=1),
        estimate_strategy_profile("zero1", dp_size=8, tp_size=1, pp_size=1),
        estimate_strategy_profile("zero2", dp_size=8, tp_size=1, pp_size=1),
        estimate_strategy_profile("fsdp", dp_size=8, tp_size=1, pp_size=1),
        estimate_strategy_profile("3d", dp_size=2, tp_size=2, pp_size=2),
    ]
