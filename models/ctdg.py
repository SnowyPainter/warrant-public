from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .attention import SinusoidalTimeEncoding, WarrantedAttention
from .base import ModelResult, WarrantModel
from .warrant_block import EdgeConditionedWarrantBlock


class _PairScorer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.scale = math.sqrt(float(hidden_dim))

    def forward(self, src_state: torch.Tensor, dst_state: torch.Tensor, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        dot = (src_state * dst_state).sum(dim=-1) / self.scale
        mlp = self.mlp(torch.cat([src_state, dst_state, query, context], dim=-1)).squeeze(-1)
        return dot + mlp


class _CTDGBase(WarrantModel):
    model_name = "ctdg_base"

    def __init__(
        self,
        *,
        num_nodes: int,
        edge_feat_dim: int = 1,
        hidden_dim: int = 128,
        time_dim: int = 32,
        use_warrant: bool = False,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__(use_warrant=use_warrant)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.time_encoder = SinusoidalTimeEncoding(time_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.memory_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attn = WarrantedAttention(
            hidden_dim,
            hidden_dim,
            use_warrant=use_warrant,
            dropout=dropout,
            gate_init=gate_init,
            gate_leak=gate_leak,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.scorer = _PairScorer(hidden_dim, dropout)
        self.edge_warrant = (
            EdgeConditionedWarrantBlock(
                hidden_dim,
                time_dim,
                dropout=dropout,
                gate_init=gate_init,
                gate_leak=gate_leak,
            )
            if use_warrant
            else None
        )

    def _encode_raw_history(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        node_h = self.node_embedding(history_nodes.long().clamp_min(0).clamp_max(self.num_nodes - 1))
        edge_h = self.edge_encoder(history_edge_feats.float())
        dt_h = self.time_encoder((query_time.unsqueeze(1) - history_times.float()).clamp_min(0.0))
        return self.memory_proj(torch.cat([node_h, edge_h, dt_h], dim=-1))

    def encode_memory(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        query_time: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._encode_raw_history(history_nodes, history_times, history_edge_feats, query_time), history_mask

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        dst_history_nodes: torch.Tensor | None = None,
        dst_history_times: torch.Tensor | None = None,
        dst_history_edge_feats: torch.Tensor | None = None,
        dst_history_mask: torch.Tensor | None = None,
    ) -> ModelResult:
        src_h = self.node_embedding(src.long().clamp_min(0).clamp_max(self.num_nodes - 1))
        dst_h = self.node_embedding(dst.long().clamp_min(0).clamp_max(self.num_nodes - 1))
        time_h = self.time_encoder(timestamp.float())
        query = self.query_proj(torch.cat([src_h, dst_h, time_h], dim=-1))
        raw_memory = self._encode_raw_history(history_nodes, history_times, history_edge_feats, timestamp)
        memory, memory_mask = self.encode_memory(history_nodes, history_times, history_edge_feats, timestamp, history_mask)
        context, info = self.attn(query, memory, mask=memory_mask)
        src_state = self.output_norm(src_h + context)
        if self.edge_warrant is not None and dst_history_nodes is not None and dst_history_times is not None:
            if dst_history_edge_feats is None:
                dst_history_edge_feats = torch.zeros_like(history_edge_feats)
            if dst_history_mask is None:
                dst_history_mask = torch.ones_like(dst_history_nodes, dtype=torch.bool)
            dst_memory = self._encode_raw_history(dst_history_nodes, dst_history_times, dst_history_edge_feats, timestamp)
            src_delta_t = (timestamp.unsqueeze(1) - history_times.float()).clamp_min(0.0)
            dst_delta_t = (timestamp.unsqueeze(1) - dst_history_times.float()).clamp_min(0.0)
            src_stats = self._edge_warrant_stats(
                center_ids=src,
                counterpart_ids=dst,
                history_nodes=history_nodes,
                delta_t=src_delta_t,
                mask=history_mask,
            )
            dst_stats = self._edge_warrant_stats(
                center_ids=dst,
                counterpart_ids=src,
                history_nodes=dst_history_nodes,
                delta_t=dst_delta_t,
                mask=dst_history_mask,
            )
            src_delta, dst_delta, edge_info = self.edge_warrant(
                src_h=src_state,
                dst_h=dst_h,
                time_h=time_h,
                src_memory=raw_memory,
                dst_memory=dst_memory,
                src_delta_t_enc=self.time_encoder(src_delta_t),
                dst_delta_t_enc=self.time_encoder(dst_delta_t),
                src_stats=src_stats,
                dst_stats=dst_stats,
                src_mask=history_mask if history_mask is not None else torch.ones_like(history_nodes, dtype=torch.bool),
                dst_mask=dst_history_mask,
            )
            src_state = self.output_norm(src_state + src_delta)
            dst_h = self.output_norm(dst_h + dst_delta)
            info.update(edge_info)
        logits = self.scorer(src_state, dst_h, query, context)
        return ModelResult(logits=logits, aux=self.aux_from_info(info))

    def _edge_warrant_stats(
        self,
        *,
        center_ids: torch.Tensor,
        counterpart_ids: torch.Tensor,
        history_nodes: torch.Tensor,
        delta_t: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = mask if mask is not None else torch.ones_like(history_nodes, dtype=torch.bool)
        counterpart_hit = (history_nodes.long() == counterpart_ids.long().unsqueeze(1)).to(delta_t.dtype) * valid.to(delta_t.dtype)
        recency = torch.exp(-torch.log1p(delta_t.float()).clamp_min(0.0) / 8.0) * counterpart_hit
        valid_ratio = valid.to(delta_t.dtype).mean(dim=1, keepdim=True).expand_as(delta_t)
        return torch.stack([counterpart_hit, recency, valid_ratio], dim=-1)


class TGAT(_CTDGBase):
    """TGAT-style temporal neighbor attention.

    The benchmark-native version keeps TGAT's important contract: query-time
    node-pair attention over source temporal neighbors with functional time
    encodings.  Warrant on/off changes the value aggregation inside that
    temporal attention path.
    """

    model_name = "TGAT"


class DyGFormer(_CTDGBase):
    """DyGFormer-style patch encoder over temporal neighbor sequences."""

    model_name = "DyGFormer"

    def __init__(
        self,
        *args,
        num_layers: int = 2,
        num_heads: int = 4,
        patch_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = max(1, int(patch_size))
        self.patch_proj = nn.Sequential(
            nn.Linear(self.patch_size * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=kwargs.get("dropout", 0.1),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.patch_transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def encode_memory(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        query_time: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        memory = self._encode_raw_history(history_nodes, history_times, history_edge_feats, query_time)
        batch_size, seq_len, hidden_dim = memory.shape
        pad = (-seq_len) % self.patch_size
        if pad:
            memory = F.pad(memory, (0, 0, pad, 0))
            if history_mask is not None:
                history_mask = F.pad(history_mask, (pad, 0), value=False)
        patches = memory.reshape(batch_size, -1, self.patch_size * hidden_dim)
        patch_memory = self.patch_proj(patches)
        if history_mask is None:
            patch_mask = None
        else:
            patch_mask = history_mask.reshape(batch_size, -1, self.patch_size).any(dim=-1)
            empty = ~patch_mask.any(dim=1)
            if bool(empty.any()):
                patch_mask = patch_mask.clone()
                patch_mask[empty, 0] = True
                patch_memory = patch_memory.clone()
                patch_memory[empty, 0] = 0.0
        src_key_padding_mask = None if patch_mask is None else ~patch_mask
        patch_memory = self.patch_transformer(patch_memory, src_key_padding_mask=src_key_padding_mask)
        return patch_memory, patch_mask


class _MixerBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.channel = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.token_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.channel(x)
        scores = self.token_gate(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            empty = ~mask.any(dim=-1, keepdim=True)
            scores = torch.where(empty, torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        global_token = (weights * x).sum(dim=1, keepdim=True)
        return x + global_token


class GraphMixer(_CTDGBase):
    """GraphMixer-style MLP mixer over fixed recent temporal tokens."""

    model_name = "GraphMixer"

    def __init__(self, *args, mixer_layers: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dropout = float(kwargs.get("dropout", 0.1))
        self.mixers = nn.ModuleList([_MixerBlock(self.hidden_dim, dropout) for _ in range(mixer_layers)])

    def encode_memory(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        query_time: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        memory = self._encode_raw_history(history_nodes, history_times, history_edge_feats, query_time)
        for mixer in self.mixers:
            memory = mixer(memory, history_mask)
        return memory, history_mask
