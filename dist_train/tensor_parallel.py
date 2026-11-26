from typing import List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_train.collectives import InMemoryGroup


class _CopyToTPRegion(torch.autograd.Function):
    """Megatron `f` operator: Identity in forward, AllReduce(sum) across TP ranks in backward."""

    @staticmethod
    def forward(ctx, comm: InMemoryGroup, *xs: torch.Tensor):
        ctx.comm = comm
        return tuple(x.clone() for x in xs)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        reduced = ctx.comm.all_reduce(list(grad_outputs), op="sum")
        return (None, *reduced)


class _ReduceFromTPRegion(torch.autograd.Function):
    """Megatron `g` operator: AllReduce(sum) across TP ranks in forward, Identity in backward."""

    @staticmethod
    def forward(ctx, comm: InMemoryGroup, *xs: torch.Tensor):
        reduced = comm.all_reduce(list(xs), op="sum")
        return tuple(reduced)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        return (None, *(g.clone() for g in grad_outputs))


def copy_to_tp_region(xs: Sequence[torch.Tensor], comm: InMemoryGroup) -> List[torch.Tensor]:
    return list(_CopyToTPRegion.apply(comm, *xs))


def reduce_from_tp_region(xs: Sequence[torch.Tensor], comm: InMemoryGroup) -> List[torch.Tensor]:
    return list(_ReduceFromTPRegion.apply(comm, *xs))


class ColumnParallelLinear(nn.Module):
    """Splits weight matrix W (out_features, in_features) along `out_features` across `tp_size`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        comm: InMemoryGroup,
        bias: bool = True,
        gather_output: bool = False,
        master_linear: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        if out_features % tp_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.out_per_partition = out_features // tp_size
        self.comm = comm
        self.gather_output = gather_output

        ref = master_linear or nn.Linear(in_features, out_features, bias=bias)
        w_chunks = ref.weight.detach().chunk(tp_size, dim=0)
        self.weights = nn.ParameterList([nn.Parameter(c.clone()) for c in w_chunks])

        if bias and ref.bias is not None:
            b_chunks = ref.bias.detach().chunk(tp_size, dim=0)
            self.biases: Optional[nn.ParameterList] = nn.ParameterList(
                [nn.Parameter(c.clone()) for c in b_chunks]
            )
        else:
            self.biases = None

    def forward(self, xs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        x_parallel = copy_to_tp_region(xs, self.comm)
        outs: List[torch.Tensor] = []
        for r in range(self.tp_size):
            b = self.biases[r] if self.biases is not None else None
            outs.append(F.linear(x_parallel[r], self.weights[r], b))

        if self.gather_output:
            full = torch.cat(outs, dim=-1)
            return [full.clone() for _ in range(self.tp_size)]
        return outs


class RowParallelLinear(nn.Module):
    """Splits weight matrix W (out_features, in_features) along `in_features` across `tp_size`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        comm: InMemoryGroup,
        bias: bool = True,
        master_linear: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        if in_features % tp_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.in_per_partition = in_features // tp_size
        self.comm = comm

        ref = master_linear or nn.Linear(in_features, out_features, bias=bias)
        w_chunks = ref.weight.detach().chunk(tp_size, dim=1)
        self.weights = nn.ParameterList([nn.Parameter(c.clone()) for c in w_chunks])

        if bias and ref.bias is not None:
            self.biases: Optional[nn.ParameterList] = nn.ParameterList(
                [nn.Parameter(ref.bias.detach().clone()) for _ in range(tp_size)]
            )
        else:
            self.biases = None

    def forward(self, xs_sharded: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        partials = [F.linear(xs_sharded[r], self.weights[r], None) for r in range(self.tp_size)]
        reduced = reduce_from_tp_region(partials, self.comm)
        if self.biases is not None:
            reduced = [reduced[r] + self.biases[r] for r in range(self.tp_size)]
        return reduced
