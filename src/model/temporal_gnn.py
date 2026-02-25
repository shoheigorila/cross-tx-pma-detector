"""Multi-scale Temporal GNN for cross-transaction attack detection.

Architecture:
  - Short-scale branch (TGN, ~100 blocks): captures immediate manipulation
  - Medium-scale branch (DyGFormer-inspired, ~1000 blocks): multi-hour attacks
  - Long-scale branch (GraphMixer-inspired, ~10000 blocks): governance attacks
  - MLP classifier on concatenated embeddings
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph_builder.features import Time2Vec


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Multi-head temporal attention over a node's temporal neighbours."""

    def __init__(self, feat_dim: int, time_dim: int, num_heads: int = 4, out_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.time2vec = Time2Vec(out_dim=time_dim)
        self.q_proj = nn.Linear(feat_dim, out_dim)
        self.k_proj = nn.Linear(feat_dim + time_dim, out_dim)
        self.v_proj = nn.Linear(feat_dim + time_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(
        self,
        query: torch.Tensor,       # (B, feat_dim)
        keys: torch.Tensor,        # (B, N, feat_dim)  neighbours
        times: torch.Tensor,       # (B, N)  relative timestamps
        mask: Optional[torch.Tensor] = None,  # (B, N) bool mask
    ) -> torch.Tensor:
        B, N, _ = keys.shape
        time_emb = self.time2vec(times)  # (B, N, time_dim)
        kv_input = torch.cat([keys, time_emb], dim=-1)  # (B, N, feat_dim + time_dim)

        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_input).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_input).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, 1, N)

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, 1, head_dim)
        out = out.transpose(1, 2).reshape(B, -1)
        return self.out_proj(out)


class NodeMemory(nn.Module):
    """GRU-based node memory module (TGN-style)."""

    def __init__(self, memory_dim: int, message_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def forward(
        self,
        memory: torch.Tensor,  # (num_nodes, memory_dim)
        node_ids: torch.Tensor,  # (batch,) node indices
        messages: torch.Tensor,  # (batch, message_dim)
    ) -> torch.Tensor:
        """Update memory for specified nodes and return updated memory."""
        mem = memory.clone()
        node_mem = memory[node_ids]  # (batch, memory_dim)
        updated = self.gru(messages, node_mem)
        mem[node_ids] = updated
        return mem


class GraphReadout(nn.Module):
    """Attention-based graph-level readout."""

    def __init__(self, node_dim: int, out_dim: int):
        super().__init__()
        self.gate = nn.Linear(node_dim, 1)
        self.proj = nn.Linear(node_dim, out_dim)

    def forward(self, node_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_embeddings: (B, N, node_dim) or (N, node_dim)
            mask: optional (B, N) bool
        Returns:
            (B, out_dim) or (out_dim,)
        """
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        gate_scores = self.gate(node_embeddings)  # (B, N, 1)
        if mask is not None:
            gate_scores = gate_scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        gate_scores = F.softmax(gate_scores, dim=1)

        projected = self.proj(node_embeddings)  # (B, N, out_dim)
        pooled = (gate_scores * projected).sum(dim=1)  # (B, out_dim)

        if squeeze:
            pooled = pooled.squeeze(0)
        return pooled


# ---------------------------------------------------------------------------
# Scale-specific branches
# ---------------------------------------------------------------------------

class ShortScaleTGN(nn.Module):
    """TGN-style model for short time windows (~100 blocks).

    Uses node memory + temporal attention for capturing immediate
    price manipulation patterns.
    """

    def __init__(
        self,
        node_feat_dim: int = 11,
        edge_feat_dim: int = 30,
        memory_dim: int = 64,
        time_dim: int = 16,
        embedding_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim

        self.node_proj = nn.Linear(node_feat_dim, memory_dim)
        self.message_fn = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_feat_dim + time_dim, memory_dim),
            nn.ReLU(),
        )
        self.memory_module = NodeMemory(memory_dim, memory_dim)
        self.temporal_attn = TemporalAttention(
            memory_dim, time_dim, num_heads, embedding_dim,
        )
        self.time2vec = Time2Vec(out_dim=time_dim)
        self.readout = GraphReadout(embedding_dim, embedding_dim)

    def forward(
        self,
        node_features: torch.Tensor,  # (num_nodes, node_feat_dim)
        sources: torch.Tensor,         # (num_edges,)
        destinations: torch.Tensor,    # (num_edges,)
        timestamps: torch.Tensor,      # (num_edges,)
        edge_features: torch.Tensor,   # (num_edges, edge_feat_dim)
    ) -> torch.Tensor:
        num_nodes = node_features.shape[0]

        # Initialize memory from node features
        memory = self.node_proj(node_features)  # (N, memory_dim)

        # Process edges chronologically
        time_emb = self.time2vec(timestamps)  # (E, time_dim)

        for i in range(len(sources)):
            src, dst = sources[i], destinations[i]
            src_mem = memory[src].unsqueeze(0)
            dst_mem = memory[dst].unsqueeze(0)
            e_feat = edge_features[i].unsqueeze(0)
            t_feat = time_emb[i].unsqueeze(0)

            msg = self.message_fn(torch.cat([src_mem, dst_mem, e_feat, t_feat], dim=-1))
            memory = self.memory_module(memory, src.unsqueeze(0), msg)
            memory = self.memory_module(memory, dst.unsqueeze(0), msg)

        return self.readout(memory)  # (embedding_dim,)


class MediumScaleDyGFormer(nn.Module):
    """Simplified DyGFormer-inspired model for medium time windows (~1000 blocks).

    Uses patch-based self-attention over temporal edge sequences.
    """

    def __init__(
        self,
        edge_feat_dim: int = 30,
        time_dim: int = 16,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(edge_feat_dim + time_dim, hidden_dim)
        self.time2vec = Time2Vec(out_dim=time_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.readout_proj = nn.Linear(hidden_dim, hidden_dim)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        timestamps: torch.Tensor,    # (num_edges,)
        edge_features: torch.Tensor,  # (num_edges, edge_feat_dim)
    ) -> torch.Tensor:
        time_emb = self.time2vec(timestamps)  # (E, time_dim)
        x = torch.cat([edge_features, time_emb], dim=-1)  # (E, feat+time)

        # Truncate/pad to max_seq_len
        if x.shape[0] > self.max_seq_len:
            # Sample uniformly
            indices = torch.linspace(0, x.shape[0] - 1, self.max_seq_len).long()
            x = x[indices]
        elif x.shape[0] < self.max_seq_len:
            pad = torch.zeros(self.max_seq_len - x.shape[0], x.shape[1], device=x.device)
            x = torch.cat([x, pad], dim=0)

        x = self.input_proj(x).unsqueeze(0)  # (1, seq, hidden)
        x = self.transformer(x)
        pooled = x.mean(dim=1)  # (1, hidden)
        return self.readout_proj(pooled.squeeze(0))  # (hidden,)


class LongScaleGraphMixer(nn.Module):
    """MLP-Mixer-inspired model for long time windows (~10000 blocks).

    Lightweight architecture suitable for longer sequences where
    full attention would be too expensive.
    """

    def __init__(
        self,
        edge_feat_dim: int = 30,
        time_dim: int = 16,
        hidden_dim: int = 64,
        num_patches: int = 32,
        num_mixer_layers: int = 2,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = edge_feat_dim + time_dim
        self.time2vec = Time2Vec(out_dim=time_dim)

        self.patch_proj = nn.Linear(self.patch_dim, hidden_dim)

        self.mixer_layers = nn.ModuleList()
        for _ in range(num_mixer_layers):
            self.mixer_layers.append(nn.ModuleDict({
                "token_mix": nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "channel_mix": nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
            }))

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        timestamps: torch.Tensor,    # (num_edges,)
        edge_features: torch.Tensor,  # (num_edges, edge_feat_dim)
    ) -> torch.Tensor:
        time_emb = self.time2vec(timestamps)
        x = torch.cat([edge_features, time_emb], dim=-1)  # (E, feat+time)

        # Create patches by splitting edge sequence
        E = x.shape[0]
        if E < self.num_patches:
            pad = torch.zeros(self.num_patches - E, x.shape[1], device=x.device)
            x = torch.cat([x, pad], dim=0)
            E = self.num_patches

        patch_size = E // self.num_patches
        patches = []
        for i in range(self.num_patches):
            start = i * patch_size
            end = start + patch_size if i < self.num_patches - 1 else E
            patches.append(x[start:end].mean(dim=0))

        x = torch.stack(patches)  # (num_patches, patch_dim)
        x = self.patch_proj(x)    # (num_patches, hidden_dim)

        for layer in self.mixer_layers:
            # Token mixing (across patches)
            x = x + layer["token_mix"](x)
            # Channel mixing (across features)
            x = x + layer["channel_mix"](x)

        pooled = x.mean(dim=0)  # (hidden_dim,)
        return self.output_proj(pooled)


# ---------------------------------------------------------------------------
# Multi-Scale Detector (main model)
# ---------------------------------------------------------------------------

class MultiScaleDetector(nn.Module):
    """Multi-scale temporal GNN that combines short/medium/long branches
    for cross-transaction price manipulation attack detection.

    Each branch operates on a different time scale:
      - Short (~100 blocks, 20 min): immediate manipulation
      - Medium (~1000 blocks, 3.3 hours): multi-step attacks
      - Long (~10000 blocks, 33 hours): governance attacks, slow rugs
    """

    def __init__(
        self,
        node_feat_dim: int = 11,
        edge_feat_dim: int = 30,
        embedding_dim: int = 64,
        time_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.short_branch = ShortScaleTGN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            memory_dim=embedding_dim,
            time_dim=time_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.medium_branch = MediumScaleDyGFormer(
            edge_feat_dim=edge_feat_dim,
            time_dim=time_dim,
            hidden_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.long_branch = LongScaleGraphMixer(
            edge_feat_dim=edge_feat_dim,
            time_dim=time_dim,
            hidden_dim=embedding_dim,
        )

        combined_dim = embedding_dim * 3  # 192

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        # Short-scale inputs
        short_node_features: torch.Tensor,
        short_sources: torch.Tensor,
        short_destinations: torch.Tensor,
        short_timestamps: torch.Tensor,
        short_edge_features: torch.Tensor,
        # Medium-scale inputs
        medium_timestamps: torch.Tensor,
        medium_edge_features: torch.Tensor,
        # Long-scale inputs
        long_timestamps: torch.Tensor,
        long_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all three branches + classifier.

        Returns:
            logits: (2,) unnormalised class scores [normal, attack]
        """
        short_emb = self.short_branch(
            short_node_features, short_sources, short_destinations,
            short_timestamps, short_edge_features,
        )
        medium_emb = self.medium_branch(medium_timestamps, medium_edge_features)
        long_emb = self.long_branch(long_timestamps, long_edge_features)

        combined = torch.cat([short_emb, medium_emb, long_emb], dim=-1)
        return self.classifier(combined)
