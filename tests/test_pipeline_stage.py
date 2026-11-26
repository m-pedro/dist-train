import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_train.pipeline import P2PChannelStats, partition_layers_into_stages


def test_pipeline_stage_partitioning_and_p2p_boundary_gradients():
    torch.manual_seed(21)
    layers = [nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16), nn.Linear(16, 6)]
    ref = nn.Sequential(*copy.deepcopy(layers))

    stages = partition_layers_into_stages(copy.deepcopy(layers), pp_size=2)
    p2p = P2PChannelStats()

    x = torch.randn(8, 16)
    y = torch.randint(0, 6, (8,))

    loss_ref = F.cross_entropy(ref(x), y)
    loss_ref.backward()

    # Stage 0 Forward -> P2P Send -> Stage 1 Forward -> Loss -> Stage 1 Backward -> P2P Send -> Stage 0 Backward
    act0 = stages[0].forward_microbatch(0, x)
    wire_act = p2p.record_send(act0)
    out1 = stages[1].forward_microbatch(0, wire_act)
    loss_pp = F.cross_entropy(out1, y)

    grad_to_0 = stages[1].backward_microbatch(0, loss_pp)
    assert grad_to_0 is not None
    wire_grad = p2p.record_send(grad_to_0)
    stages[0].backward_microbatch(0, wire_grad)

    assert p2p.p2p_messages == 2 and p2p.p2p_bytes > 0
    assert stages[0].current_inflight == 0 and stages[1].current_inflight == 0

    pp_params = [p for st in stages for p in st.module.parameters()]
    for p_r, p_p in zip(ref.parameters(), pp_params):
        assert p_r.grad is not None and p_p.grad is not None
        assert torch.allclose(p_r.grad, p_p.grad, atol=1e-5)
