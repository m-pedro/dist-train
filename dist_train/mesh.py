from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class ParallelCoordinates:
    dp_rank: int
    pp_rank: int
    tp_rank: int


class DeviceMesh3D:
    """Maps flat global ranks [0 .. world_size - 1] onto a 3D (DP, PP, TP) cluster topology."""

    def __init__(self, dp_size: int = 1, pp_size: int = 1, tp_size: int = 1) -> None:
        if dp_size <= 0 or pp_size <= 0 or tp_size <= 0:
            raise ValueError("dp_size, pp_size, and tp_size must all be >= 1")
        self.dp_size = dp_size
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.world_size = dp_size * pp_size * tp_size
        self.grid = np.arange(self.world_size, dtype=np.int64).reshape(
            (dp_size, pp_size, tp_size)
        )

    def coords_of(self, global_rank: int) -> ParallelCoordinates:
        if not (0 <= global_rank < self.world_size):
            raise ValueError(f"global_rank {global_rank} out of [0, {self.world_size})")
        dp_r, pp_r, tp_r = np.unravel_index(global_rank, (self.dp_size, self.pp_size, self.tp_size))
        return ParallelCoordinates(dp_rank=int(dp_r), pp_rank=int(pp_r), tp_rank=int(tp_r))

    def get_tp_group(self, global_rank: int) -> List[int]:
        c = self.coords_of(global_rank)
        return [int(r) for r in self.grid[c.dp_rank, c.pp_rank, :]]

    def get_pp_group(self, global_rank: int) -> List[int]:
        c = self.coords_of(global_rank)
        return [int(r) for r in self.grid[c.dp_rank, :, c.tp_rank]]

    def get_dp_group(self, global_rank: int) -> List[int]:
        c = self.coords_of(global_rank)
        return [int(r) for r in self.grid[:, c.pp_rank, c.tp_rank]]

    def shape(self) -> Tuple[int, int, int]:
        return (self.dp_size, self.pp_size, self.tp_size)
