"""Network assembly.

    observations -> node features
                 -> structural encoder over (A^II, A^IM)      equations 3-5
                 -> gate temporal encoder                      equations 6-9
                 -> context vector h_T
                 -> decoder over the candidate set             equations 10-12
                 -> intervention order pi

A regression head on [h_T, pooled embedding] estimates the recovery node; it is
an independent output of the same encoder and does not affect the ordering
objective.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .graph_layers import AttentionAggregation, HomogeneousPassing, TypedMessagePassing
from .temporal import TemporalEncoder
from .decoders import PointerDecoder, Seq2SeqDecoder, SoftmaxDecoder


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 32):
        super().__init__()
        table = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        table[:, 0::2] = torch.sin(position * div)
        table[:, 1::2] = torch.cos(position * div[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table[:, : x.shape[1], :]


ENCODERS = ("typed", "homo", "gat", "plain", "transformer")


class SequencingModel(nn.Module):
    def __init__(self, config, encoder: str = "typed", decoder: str = "pointer",
                 use_ii: bool = True, use_im: bool = True,
                 shared_weight: bool = False):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder: {encoder}")

        self.config = config
        self.encoder_kind = encoder
        self.use_ii, self.use_im = use_ii, use_im
        self.n_interventions = config.n_interventions
        self.n_nodes = config.n_interventions + config.n_indicators

        d, d_prime, hidden = config.feature_dim, config.gcn_dim, config.hidden_dim

        if encoder in ("typed", "homo", "gat"):
            if encoder == "typed":
                self.graph = TypedMessagePassing(d, d_prime, config.dropout,
                                                 shared_weight=shared_weight)
            elif encoder == "homo":
                self.graph = HomogeneousPassing(d, d_prime, config.dropout)
            else:
                self.graph = AttentionAggregation(d, d_prime, config.dropout)
            temporal_input = config.n_interventions * d_prime
        else:
            self.graph = None
            temporal_input = self.n_nodes * d

        self.projector = nn.Linear(self.n_nodes * d, config.n_interventions * d_prime)

        if encoder == "transformer":
            self.position = PositionalEncoding(hidden)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                                           dim_feedforward=hidden * 2,
                                           dropout=config.dropout, batch_first=True),
                num_layers=1)
            self.input_proj = nn.Linear(self.n_nodes * d, hidden)
            self.temporal = None
        else:
            self.transformer = None
            self.temporal = TemporalEncoder(temporal_input, hidden)

        self.decoder_head = self._build_decoder(decoder, d_prime, hidden)
        self.flatus_head = nn.Sequential(
            nn.Linear(hidden + d_prime, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.contrib_trace: list = []

    def _build_decoder(self, kind: str, d_prime: int, hidden: int):
        builders = {
            "pointer": lambda: PointerDecoder(d_prime, hidden, self.config.pointer_dim),
            "softmax": lambda: SoftmaxDecoder(d_prime, hidden, self.n_interventions),
            "seq2seq": lambda: Seq2SeqDecoder(d_prime, hidden, self.n_interventions),
        }
        if kind not in builders:
            raise ValueError(f"unknown decoder: {kind}")
        return builders[kind]()

    def encode(self, X, AI, AM, mask):
        batch, steps, _, _ = X.shape
        self.contrib_trace = []
        device = X.device

        if self.graph is not None:
            embeddings = []
            for t in range(steps):
                a_ii = AI[:, t] if self.use_ii else torch.zeros_like(AI[:, t])
                a_im = AM[:, t] if self.use_im else torch.zeros_like(AM[:, t])
                embeddings.append(self.graph(X[:, t], a_ii, a_im, self.n_interventions))
                self.contrib_trace.append(dict(self.graph.last_contrib))
            embedding_seq = torch.stack(embeddings, dim=1)
            flat = embedding_seq.reshape(batch, steps, -1)
        else:
            flat = X.reshape(batch, steps, -1)
            embedding_seq = self.projector(flat).reshape(batch, steps, self.n_interventions, -1)

        if self.transformer is not None:
            hidden_seq = self.transformer(self.position(self.input_proj(flat)),
                                          src_key_padding_mask=~mask)
            context = hidden_seq[torch.arange(batch, device=device), mask.sum(dim=1) - 1]
            gate_trace = []
        else:
            context, gate_trace = self.temporal(flat, mask)

        last = (mask.sum(dim=1) - 1).long()
        node_embedding = embedding_seq[torch.arange(batch, device=device), last]
        return context, node_embedding, gate_trace

    def forward(self, X, AI, AM, mask, reference: Optional[torch.Tensor] = None,
                sample: bool = False, teacher_forcing: bool = True) -> Dict[str, torch.Tensor]:
        context, node_embedding, gate_trace = self.encode(X, AI, AM, mask)
        target = reference if (reference is not None and teacher_forcing) else None
        decoded = self.decoder_head(node_embedding, context, target, sample)

        raw = self.flatus_head(torch.cat([context, node_embedding.mean(dim=1)], dim=-1)).squeeze(-1)
        return {
            "log_prob": decoded["log_prob"],
            "permutation": decoded["permutation"],
            "attention": decoded["attention"],
            "context": context,
            "node_embedding": node_embedding,
            "gate_trace": gate_trace,
            "flatus": 12.0 + 76.0 * torch.sigmoid(raw),
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
