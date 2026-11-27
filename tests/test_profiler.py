from dist_train.profiler import compare_7b_cluster_strategies


def test_cluster_profiler_monotonic_zero_savings_and_fsdp_overhead_ratio():
    profiles = {p.strategy: p for p in compare_7b_cluster_strategies()}

    # 1. Standard DDP on 7B requires > 100 GB per GPU (OOMs on 80 GB H100), while ZeRO/FSDP/3D fit!
    assert not profiles["ddp"].fits_80gb_h100()
    assert profiles["zero1"].fits_80gb_h100()
    assert profiles["zero2"].fits_80gb_h100()
    assert profiles["fsdp"].fits_80gb_h100()
    assert profiles["3d"].fits_80gb_h100()

    # 2. Strict monotonic VRAM reduction: DDP > ZeRO-1 > ZeRO-2 > FSDP (ZeRO-3)
    assert profiles["ddp"].total_vram_gb > profiles["zero1"].total_vram_gb
    assert profiles["zero1"].total_vram_gb > profiles["zero2"].total_vram_gb
    assert profiles["zero2"].total_vram_gb > profiles["fsdp"].total_vram_gb

    # 3. Communication theorem: ZeRO-2 moves the EXACT same bytes as DDP, while FSDP moves 1.5x!
    assert abs(profiles["zero2"].comm_gb_per_step - profiles["ddp"].comm_gb_per_step) < 1e-6
    ratio = profiles["fsdp"].comm_gb_per_step / profiles["ddp"].comm_gb_per_step
    assert abs(ratio - 1.5) < 1e-4
