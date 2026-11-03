"""Structural encoders over the intervention graph.

    h_i,II^t = sigma( W_II * sum_j a_ij^II(t) x_j^I(t) )        (3)
    h_i,IM^t = sigma( W_IM * sum_k a_ik^IM(t) x_k^M(t) )        (4)
    e_i^t    = LayerNorm( h_i,II^t + h_i,IM^t )                 (5)

Three variants share one calling signature so that they can be swapped by
configuration: the typed encoder above, a homogeneous encoder over the combined
node set, and an attention encoder over intervention nodes only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgePath(nn.Module):
    """One edge-type path."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, node_feat: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.einsum("bij,bjd->bid", adjacency, node_feat)
        return self.dropout(torch.relu(self.linear(message)))


class TypedMessagePassing(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 shared_weight: bool = False):
        super().__init__()
        self.shared = shared_weight
        self.path_ii = EdgePath(in_dim, out_dim, dropout)
        self.path_im = None if shared_weight else EdgePath(in_dim, out_dim, dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.last_contrib = {}

    def forward(self, x, a_ii, a_im, n_interventions):
        x_i, x_m = x[:, :n_interventions, :], x[:, n_interventions:, :]
        h_ii = self.path_ii(x_i, a_ii)
        h_im = self.path_ii(x_m, a_im) if self.shared else self.path_im(x_m, a_im)
        self.last_contrib = {
            "dependency": float(h_ii.detach().abs().mean()),
            "temporal": float(h_im.detach().abs().mean()),
        }
        return self.norm(h_ii + h_im)


class HomogeneousPassing(nn.Module):
    """Single edge type over the combined node set."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.last_contrib = {}

    @staticmethod
    def combine(a_ii, a_im):
        batch, n_int, _ = a_ii.shape
        n_ind = a_im.shape[-1]
        full = a_ii.new_zeros((batch, n_int + n_ind, n_int + n_ind))
        full[:, :n_int, :n_int] = a_ii
        full[:, :n_int, n_int:] = a_im
        return full

    def forward(self, x, a_ii, a_im, n_interventions):
        out = self.dropout(torch.relu(
            self.linear(torch.einsum("bij,bjd->bid", self.combine(a_ii, a_im), x))))
        self.last_contrib = {"homogeneous": float(out.detach().abs().mean())}
        return self.norm(out)[:, :n_interventions, :]


class AttentionAggregation(nn.Module):
    """Attention-weighted aggregation over intervention neighbours."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.score = nn.Linear(2 * out_dim, 1, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.last_contrib = {}

    def forward(self, x, a_ii, a_im, n_interventions):
        proj = self.proj(x[:, :n_interventions, :])
        rows = proj.shape[1]
        left = proj.unsqueeze(2).expand(-1, -1, rows, -1)
        right = proj.unsqueeze(1).expand(-1, rows, -1, -1)
        logits = self.score(torch.cat([left, right], dim=-1)).squeeze(-1)
        weight = torch.softmax(logits.masked_fill(a_ii <= 0, -1e9), dim=-1)
        out = torch.relu(torch.einsum("bij,bjd->bid", weight, proj))
        self.last_contrib = {"attention": float(out.detach().abs().mean())}
        return self.norm(self.dropout(out))
