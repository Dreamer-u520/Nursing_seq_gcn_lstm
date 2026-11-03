"""Gated temporal encoder.

    [f_t, u_t, o_t] = sigma( W_g [e_flat^t ; h_{t-1}] + b_g )   (6)
    c_tilde_t       = tanh( W_c [e_flat^t ; h_{t-1}] + b_c )    (7)
    c_t               = f_t * c_{t-1} + u_t * c_tilde_t          (8)
    h_t               = o_t * tanh(c_t)                          (9)

Gates are returned with the hidden state so that their trajectories can be
inspected without re-deriving them.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class GatedCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = nn.Linear(input_dim + hidden_dim, 3 * hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        hidden_prev, cell_prev = state
        joined = torch.cat([x, hidden_prev], dim=-1)
        forget, write, output = torch.sigmoid(self.gate(joined)).chunk(3, dim=-1)
        candidate = torch.tanh(self.candidate(joined))
        cell = forget * cell_prev + write * candidate
        return output * torch.tanh(cell), cell, {
            "forget": forget, "write": write, "output": output, "candidate": candidate,
        }


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = GatedCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor):
        batch, steps, _ = sequence.shape
        hidden = torch.zeros(batch, self.hidden_dim, device=sequence.device)
        cell = torch.zeros(batch, self.hidden_dim, device=sequence.device)

        gate_trace = []
        for t in range(steps):
            new_hidden, new_cell, trace = self.cell(sequence[:, t], (hidden, cell))
            keep = mask[:, t].unsqueeze(-1)
            hidden = torch.where(keep, new_hidden, hidden)
            cell = torch.where(keep, new_cell, cell)
            gate_trace.append({k: v.mean(dim=-1) for k, v in trace.items()})

        return hidden, gate_trace
