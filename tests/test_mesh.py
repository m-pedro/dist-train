import pytest
from dist_train.mesh import DeviceMesh3D


def test_3d_device_mesh_orthogonal_process_groups():
    # 8-rank cluster organized as DP=2, PP=2, TP=2
    mesh = DeviceMesh3D(dp_size=2, pp_size=2, tp_size=2)
    assert mesh.world_size == 8

    # Innermost contiguous ranks form NVLink Tensor-Parallel pairs
    assert mesh.get_tp_group(0) == [0, 1]
    assert mesh.get_tp_group(5) == [4, 5]

    # Middle axis forms Pipeline-Parallel stage chains
    assert mesh.get_pp_group(0) == [0, 2]
    assert mesh.get_pp_group(1) == [1, 3]

    # Outermost axis forms Data-Parallel / FSDP replica groups
    assert mesh.get_dp_group(0) == [0, 4]
    assert mesh.get_dp_group(3) == [3, 7]

    with pytest.raises(ValueError, match=">= 1"):
        DeviceMesh3D(dp_size=0, pp_size=2, tp_size=2)
