import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_train.pipeline import partition_layers_into_stages
from dist_train.pipeline_schedule import run_1f1b_step, run_gpipe_step


def test_1f1b_matches_gpipe_and_single_device_while_slashing_inflight_activations():
    torch.manual_seed(303)
    pp_size = 4
    num_mb = 12

    base_layers = [
        nn.Sequential(nn.Linear(16, 32), nn.GELU()),
        nn.Sequential(nn.Linear(32, 32), nn.GELU()),
        nn.Sequential(nn.Linear(32, 16), nn.GELU()),
        nn.Linear(16, 8),
    ]
    ref_model = nn.Sequential(*copy.deepcopy(base_layers))
    gpipe_stages = partition_layers_into_stages(copy.deepcopy(base_layers), pp_size=pp_size)
    onef1b_stages = partition_layers_into_stages(copy.deepcopy(base_layers), pp_size=pp_size)

    x_full = torch.randn(24, 16)
    y_full = torch.randint(0, 8, (24,))

    loss_ref = F.cross_entropy(ref_model(x_full), y_full)
    loss_ref.backward()

    rep_gpipe = run_gpipe_step(
        gpipe_stages, x_full.chunk(num_mb), y_full.chunk(num_mb), F.cross_entropy
    )
    rep_1f1b = run_1f1b_step(
        onef1b_stages, x_full.chunk(num_mb), y_full.chunk(num_mb), F.cross_entropy
    )

    # 1. Exact loss and gradient parity across Single-Device, GPipe, and 1F1B!
    assert abs(rep_gpipe.mean_loss - float(loss_ref.item())) < 1e-5
    assert abs(rep_1f1b.mean_loss - float(loss_ref.item())) < 1e-5

    gpipe_params = [p for st in gpipe_stages for p in st.module.parameters()]
    onef1b_params = [p for st in onef1b_stages for p in st.module.parameters()]
    for p_ref, p_gp, p_1f in zip(ref_model.parameters(), gpipe_params, onef1b_params):
        assert p_ref.grad is not None and p_gp.grad is not None and p_1f.grad is not None
        assert torch.allclose(p_ref.grad, p_gp.grad, atol=1e-5)
        assert torch.allclose(p_ref.grad, p_1f.grad, atol=1e-5)

    # 2. Key Systems Theorem: GPipe holds M=12 in-flight activations on every stage,
    # whereas 1F1B caps in-flight activations at <= pp_size (4) on Stage 0 (and 1 on Stage 3)!
    assert rep_gpipe.peak_inflight_per_stage == [12, 12, 12, 12]
    assert max(rep_1f1b.peak_inflight_per_stage) <= pp_size
    assert rep_1f1b.peak_inflight_per_stage[-1] == 1
