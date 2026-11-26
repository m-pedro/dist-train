import torch
from dist_train.collectives import InMemoryGroup
from dist_train.tp_transformer import ReferenceTransformerBlock, TensorParallelTransformerBlock


def test_tensor_parallel_transformer_block_matches_dense_block_forward_and_backward():
    torch.manual_seed(101)
    tp_size = 4
    comm = InMemoryGroup(world_size=tp_size)

    ref_blk = ReferenceTransformerBlock(dim=32, n_heads=4, hidden_dim=64)
    tp_blk = TensorParallelTransformerBlock(
        dim=32, n_heads=4, hidden_dim=64, tp_size=tp_size, comm=comm, master_block=ref_blk
    )

    x_ref = torch.randn(3, 8, 32, requires_grad=True)
    out_ref = ref_blk(x_ref)
    (out_ref.pow(2).mean()).backward()

    xs_tp = [x_ref.detach().clone().requires_grad_(True) for _ in range(tp_size)]
    outs_tp = tp_blk(xs_tp)
    sum(o.pow(2).mean() for o in outs_tp).backward()

    for r in range(tp_size):
        assert torch.allclose(outs_tp[r], out_ref, atol=1e-5)
        assert xs_tp[r].grad is not None and x_ref.grad is not None
        assert torch.allclose(xs_tp[r].grad, x_ref.grad, atol=1e-5)

    w_gate_tp_grad = torch.cat([p.grad for p in tp_blk.w_gate.weights], dim=0)
    w_down_tp_grad = torch.cat([p.grad for p in tp_blk.w_down.weights], dim=1)
    assert ref_blk.w_gate.weight.grad is not None and ref_blk.w_down.weight.grad is not None
    assert torch.allclose(w_gate_tp_grad, ref_blk.w_gate.weight.grad, atol=1e-5)
    assert torch.allclose(w_down_tp_grad, ref_blk.w_down.weight.grad, atol=1e-5)
