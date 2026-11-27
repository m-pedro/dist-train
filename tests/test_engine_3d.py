import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_train.engine_3d import Parallel3DTrainer
from dist_train.mesh import DeviceMesh3D
from dist_train.tp_transformer import ReferenceTransformerBlock


def test_8_worker_3d_parallelism_dp2_pp2_tp2_matches_single_device_adamw():
    torch.manual_seed(2026)
    blocks = [ReferenceTransformerBlock(dim=32, n_heads=4, hidden_dim=64) for _ in range(4)]

    # Single-device reference model + AdamW
    ref_model = nn.Sequential(*copy.deepcopy(blocks))
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=5e-3, weight_decay=0.01)

    # 8-rank 3D cluster: DP=2 x PP=2 x TP=2 (with M=4 microbatches per DP rank)
    mesh = DeviceMesh3D(dp_size=2, pp_size=2, tp_size=2)
    trainer_3d = Parallel3DTrainer(
        copy.deepcopy(blocks), mesh=mesh, num_microbatches=4, lr=5e-3, weight_decay=0.01
    )

    for _ in range(2):
        x = torch.randn(16, 6, 32)
        target = torch.randn(16, 6, 32)

        ref_opt.zero_grad()
        loss_ref = F.mse_loss(ref_model(x), target)
        loss_ref.backward()
        ref_opt.step()

        metrics = trainer_3d.step_batch(x, target, F.mse_loss)
        assert abs(metrics.loss - float(loss_ref.item())) < 1e-5
        assert metrics.world_size == 8
        assert metrics.peak_inflight_stage0 <= mesh.pp_size

    # Export reconstructed 3D weights and verify exact parameter parity against Single-GPU AdamW!
    exported_blocks = trainer_3d.export_dense_blocks()
    exported_seq = nn.Sequential(*exported_blocks)
    for (name_r, p_r), (name_3d, p_3d) in zip(
        ref_model.named_parameters(), exported_seq.named_parameters()
    ):
        assert torch.allclose(p_r, p_3d, atol=1e-4), f"Mismatch in {name_r} vs {name_3d}"
