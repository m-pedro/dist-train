import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_train.collectives import InMemoryGroup
from dist_train.tensor_parallel import ColumnParallelLinear, RowParallelLinear


def test_column_and_row_parallel_mlp_exact_forward_and_backward_parity():
    torch.manual_seed(42)
    tp_size = 4
    comm = InMemoryGroup(world_size=tp_size)

    fc1_ref = nn.Linear(16, 32, bias=True)
    fc2_ref = nn.Linear(32, 16, bias=True)

    col_fc1 = ColumnParallelLinear(16, 32, tp_size=tp_size, comm=comm, master_linear=fc1_ref)
    row_fc2 = RowParallelLinear(32, 16, tp_size=tp_size, comm=comm, master_linear=fc2_ref)

    x_ref = torch.randn(6, 16, requires_grad=True)
    y_ref = fc2_ref(F.gelu(fc1_ref(x_ref)))
    y_ref.sum().backward()

    # Each TP rank runs backward on its local loss simultaneously
    xs_tp = [x_ref.detach().clone().requires_grad_(True) for _ in range(tp_size)]
    h_shards = [F.gelu(h) for h in col_fc1(xs_tp)]
    ys_tp = row_fc2(h_shards)
    sum(y.sum() for y in ys_tp).backward()

    for r in range(tp_size):
        assert torch.allclose(ys_tp[r], y_ref, atol=1e-5)
        assert xs_tp[r].grad is not None and x_ref.grad is not None
        assert torch.allclose(xs_tp[r].grad, x_ref.grad, atol=1e-5)

    col_w_grad = torch.cat([p.grad for p in col_fc1.weights], dim=0)
    row_w_grad = torch.cat([p.grad for p in row_fc2.weights], dim=1)
    assert fc1_ref.weight.grad is not None and fc2_ref.weight.grad is not None
    assert torch.allclose(col_w_grad, fc1_ref.weight.grad, atol=1e-5)
    assert torch.allclose(row_w_grad, fc2_ref.weight.grad, atol=1e-5)
