import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_train.checkpointing import CheckpointedSequential


def test_selective_activation_checkpointing_saves_memory_with_exact_gradients():
    torch.manual_seed(55)
    raw_layers = [nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)) for _ in range(8)]

    eager_model = nn.Sequential(*copy.deepcopy(raw_layers))
    ckpt_model = CheckpointedSequential(copy.deepcopy(raw_layers), checkpoint_every=2)

    # Track live tensor storage bytes allocated in PyTorch's autograd saved-tensor hooks!
    def _measure_saved_activation_bytes(fn) -> int:
        saved_bytes = {"total": 0}

        def pack_hook(t: torch.Tensor):
            saved_bytes["total"] += t.numel() * t.element_size()
            return t

        def unpack_hook(t: torch.Tensor):
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out = fn()
        return saved_bytes["total"], out

    x1 = torch.randn(12, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    targets = torch.randn(12, 32)

    eager_bytes, out_eager = _measure_saved_activation_bytes(lambda: eager_model(x1))
    ckpt_bytes, out_ckpt = _measure_saved_activation_bytes(lambda: ckpt_model(x2))

    # 1. Checkpointing saves strictly > 60% of intermediate saved activation memory!
    assert ckpt_bytes < eager_bytes * 0.40

    # 2. Forward outputs and ALL backward parameter + input gradients match identically!
    assert torch.allclose(out_eager, out_ckpt, atol=1e-6)
    F.mse_loss(out_eager, targets).backward()
    F.mse_loss(out_ckpt, targets).backward()

    assert x1.grad is not None and x2.grad is not None
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    for p_e, p_c in zip(eager_model.parameters(), ckpt_model.parameters()):
        assert p_e.grad is not None and p_c.grad is not None
        assert torch.allclose(p_e.grad, p_c.grad, atol=1e-5)
