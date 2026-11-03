"""Decoders.

    u_ik         = v_a^T tanh( W_d d_k + W_e e_i^T )            (10)
    p(pi_k | ..) = softmax over the remaining candidate set      (11)
    d_{k+1}      = GRU( e_{pi_k}^T , d_k ),   d_1 = h_T            (12)

The pointer decoder enforces the set constraint through dynamic masking. Two
control decoders reproduce the alternatives compared in the decision module
experiment: a position-wise classifier without inter-step memory, and an
attention decoder without the set constraint.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class PointerDecoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, pointer_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, pointer_dim)
        self.key = nn.Linear(embed_dim, pointer_dim)
        self.score = nn.Linear(pointer_dim, 1, bias=False)
        self.cell = nn.GRUCell(embed_dim, hidden_dim)
        self.embed_dim = embed_dim

    def forward(self, node_embedding: torch.Tensor, context: torch.Tensor,
                reference: Optional[torch.Tensor] = None,
                sample: bool = False) -> Dict[str, torch.Tensor]:
        batch, n_items, _ = node_embedding.shape
        hidden = context
        selected = torch.zeros(batch, n_items, dtype=torch.bool, device=node_embedding.device)
        log_probs, permutations, attention = [], [], []

        for step in range(n_items):
            logits = self.score(torch.tanh(
                self.query(hidden).unsqueeze(1) + self.key(node_embedding)
            )).squeeze(-1)
            distribution = torch.log_softmax(logits.masked_fill(selected, -1e9), dim=-1)
            attention.append(distribution.detach())

            if reference is not None:
                index = reference[:, step]
            elif sample:
                index = torch.multinomial(distribution.exp(), 1).squeeze(-1)
            else:
                index = distribution.argmax(dim=-1)

            log_probs.append(distribution.gather(1, index.unsqueeze(1)).squeeze(1))
            permutations.append(index)
            selected = selected.scatter(1, index.unsqueeze(1), True)

            chosen = node_embedding.gather(
                1, index.view(batch, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(1)
            hidden = self.cell(chosen, hidden)

        return {
            "log_prob": torch.stack(log_probs, dim=1),
            "permutation": torch.stack(permutations, dim=1),
            "attention": torch.stack(attention, dim=1),
            "final_state": hidden,
        }


class SoftmaxDecoder(nn.Module):
    """Position-wise multi-class scoring without inter-step memory."""

    def __init__(self, embed_dim: int, hidden_dim: int, n_items: int):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(n_items, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2 + embed_dim, 1)

    def forward(self, node_embedding: torch.Tensor, context: torch.Tensor,
                reference: Optional[torch.Tensor] = None,
                sample: bool = False) -> Dict[str, torch.Tensor]:
        batch, n_items, _ = node_embedding.shape
        context_repr = self.query(context).view(batch, 1, 1, -1).expand(-1, n_items, n_items, -1)
        position = self.position.view(1, n_items, 1, -1).expand(batch, -1, n_items, -1)
        node = node_embedding.unsqueeze(1).expand(-1, n_items, -1, -1)

        logits = self.proj(torch.cat([context_repr, position, node], dim=-1)).squeeze(-1)
        distribution = torch.log_softmax(logits, dim=-1)

        # Independent heads can repeat an item, so the scores are resolved into a
        # valid assignment by maximising total position-item score.
        from scipy.optimize import linear_sum_assignment

        scores = logits.detach().cpu().numpy()
        permutation_np = np.stack([
            linear_sum_assignment(-scores[b])[1] for b in range(batch)
        ]).astype(np.int64)
        permutation = torch.from_numpy(permutation_np).to(logits.device)

        gather_index = (reference if reference is not None else permutation).unsqueeze(2)
        return {
            "log_prob": distribution.gather(2, gather_index).squeeze(-1),
            "permutation": permutation,
            "attention": distribution.detach().mean(dim=1),
            "final_state": context,
        }


class Seq2SeqDecoder(nn.Module):
    """Attention decoder over the candidate set without masking."""

    def __init__(self, embed_dim: int, hidden_dim: int, n_items: int):
        super().__init__()
        self.cell = nn.GRUCell(embed_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim + embed_dim, n_items)
        self.start = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.start, std=0.02)

    def forward(self, node_embedding: torch.Tensor, context: torch.Tensor,
                reference: Optional[torch.Tensor] = None,
                sample: bool = False) -> Dict[str, torch.Tensor]:
        batch, n_items, _ = node_embedding.shape
        hidden = context
        previous = self.start.expand(batch, -1)
        key = self.key(node_embedding)

        log_probs, permutations, attention = [], [], []
        for step in range(n_items):
            hidden = self.cell(previous, hidden)
            weight = torch.softmax(
                torch.tanh(self.query(hidden).unsqueeze(1) + key).mean(dim=-1), dim=-1)
            attended = torch.einsum("bi,bie->be", weight, node_embedding)
            distribution = torch.log_softmax(
                self.output(torch.cat([hidden, attended], dim=-1)), dim=-1)
            attention.append(weight.detach())

            if reference is not None:
                index = reference[:, step]
            elif sample:
                index = torch.multinomial(distribution.exp(), 1).squeeze(-1)
            else:
                index = distribution.argmax(dim=-1)

            log_probs.append(distribution.gather(1, index.unsqueeze(1)).squeeze(1))
            permutations.append(index)
            previous = node_embedding.gather(
                1, index.view(batch, 1, 1).expand(-1, 1, node_embedding.shape[-1])).squeeze(1)

        return {
            "log_prob": torch.stack(log_probs, dim=1),
            "permutation": torch.stack(permutations, dim=1),
            "attention": torch.stack(attention, dim=1),
            "final_state": hidden,
        }
