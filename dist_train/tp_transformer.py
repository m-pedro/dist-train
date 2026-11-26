import math
from typing import List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_train.collectives import InMemoryGroup
from dist_train.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class ReferenceTransformerBlock(nn.Module):
    """Single-device dense Transformer block (RMSNorm + Causal MHA + SwiGLU) for exact parity checks."""

    def __init__(self, dim: int, n_heads: int, hidden_dim: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.hidden_dim = hidden_dim

        self.attn_norm_w = nn.Parameter(torch.ones(dim))
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.ffn_norm_w = nn.Parameter(torch.ones(dim))
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    @staticmethod
    def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x * rms * weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h_norm = self._rms_norm(x, self.attn_norm_w)

        q = self.wq(h_norm).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h_norm).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h_norm).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        probs = F.softmax(scores + mask, dim=-1)
        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        x = x + self.wo(attn_out)

        f_norm = self._rms_norm(x, self.ffn_norm_w)
        x = x + self.w_down(F.silu(self.w_gate(f_norm)) * self.w_up(f_norm))
        return x


class TensorParallelTransformerBlock(nn.Module):
    """Megatron-LM Tensor-Parallel Transformer block sharding Attention Heads and SwiGLU across `tp_size`."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        hidden_dim: int,
        tp_size: int,
        comm: InMemoryGroup,
        master_block: Optional[ReferenceTransformerBlock] = None,
    ) -> None:
        super().__init__()
        if n_heads % tp_size != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by tp_size ({tp_size})")
        if hidden_dim % tp_size != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by tp_size ({tp_size})")

        self.dim = dim
        self.n_heads = n_heads
        self.tp_size = tp_size
        self.heads_per_rank = n_heads // tp_size
        self.head_dim = dim // n_heads
        self.comm = comm

        ref = master_block or ReferenceTransformerBlock(dim, n_heads, hidden_dim)

        self.attn_norm_ws = nn.ParameterList(
            [nn.Parameter(ref.attn_norm_w.detach().clone()) for _ in range(tp_size)]
        )
        self.wq = ColumnParallelLinear(dim, dim, tp_size, comm, bias=False, master_linear=ref.wq)
        self.wk = ColumnParallelLinear(dim, dim, tp_size, comm, bias=False, master_linear=ref.wk)
        self.wv = ColumnParallelLinear(dim, dim, tp_size, comm, bias=False, master_linear=ref.wv)
        self.wo = RowParallelLinear(dim, dim, tp_size, comm, bias=False, master_linear=ref.wo)

        self.ffn_norm_ws = nn.ParameterList(
            [nn.Parameter(ref.ffn_norm_w.detach().clone()) for _ in range(tp_size)]
        )
        self.w_gate = ColumnParallelLinear(
            dim, hidden_dim, tp_size, comm, bias=False, master_linear=ref.w_gate
        )
        self.w_up = ColumnParallelLinear(
            dim, hidden_dim, tp_size, comm, bias=False, master_linear=ref.w_up
        )
        self.w_down = RowParallelLinear(
            hidden_dim, dim, tp_size, comm, bias=False, master_linear=ref.w_down
        )

    def forward(self, xs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        bsz, seq_len, _ = xs[0].shape
        h_norms = [
            ReferenceTransformerBlock._rms_norm(xs[r], self.attn_norm_ws[r])
            for r in range(self.tp_size)
        ]

        q_shards = self.wq(h_norms)
        k_shards = self.wk(h_norms)
        v_shards = self.wv(h_norms)

        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=xs[0].device, dtype=xs[0].dtype),
            diagonal=1,
        )

        head_outs: List[torch.Tensor] = []
        for r in range(self.tp_size):
            qr = q_shards[r].view(bsz, seq_len, self.heads_per_rank, self.head_dim).transpose(1, 2)
            kr = k_shards[r].view(bsz, seq_len, self.heads_per_rank, self.head_dim).transpose(1, 2)
            vr = v_shards[r].view(bsz, seq_len, self.heads_per_rank, self.head_dim).transpose(1, 2)

            scores = torch.matmul(qr, kr.transpose(-2, -1)) / math.sqrt(self.head_dim)
            probs = F.softmax(scores + mask, dim=-1)
            out_r = (
                torch.matmul(probs, vr)
                .transpose(1, 2)
                .contiguous()
                .view(bsz, seq_len, self.heads_per_rank * self.head_dim)
            )
            head_outs.append(out_r)

        # Single AllReduce across TP ranks via RowParallelLinear `wo`!
        attn_proj = self.wo(head_outs)
        xs_mid = [xs[r] + attn_proj[r] for r in range(self.tp_size)]

        f_norms = [
            ReferenceTransformerBlock._rms_norm(xs_mid[r], self.ffn_norm_ws[r])
            for r in range(self.tp_size)
        ]
        gate_shards = self.w_gate(f_norms)
        up_shards = self.w_up(f_norms)
        swiglu_shards = [F.silu(gate_shards[r]) * up_shards[r] for r in range(self.tp_size)]

        # Second AllReduce across TP ranks via RowParallelLinear `w_down`!
        ffn_proj = self.w_down(swiglu_shards)
        return [xs_mid[r] + ffn_proj[r] for r in range(self.tp_size)]
