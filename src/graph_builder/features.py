"""Feature computation for temporal graph nodes and edges."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Learnable time encoding: v(t) = [t, sin(w_1*t + phi_1), ...]

    Reference: Kazemi et al., "Time2Vec: Learning a General Representation of Time" (2019)
    """

    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim
        # First element is linear; rest are periodic
        self.w = nn.Parameter(torch.randn(out_dim - 1))
        self.b = nn.Parameter(torch.randn(out_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (...,) – timestamps (seconds or block numbers).
        Returns:
            Tensor of shape (..., out_dim).
        """
        t = t.unsqueeze(-1).float()  # (..., 1)
        linear = t  # (..., 1)
        periodic = torch.sin(t * self.w + self.b)  # (..., out_dim - 1)
        return torch.cat([linear, periodic], dim=-1)


def compute_node_features(
    address: str,
    events: list,  # list of UnifiedEvent in the window
    known_dex: set[str],
    known_lending: set[str],
) -> np.ndarray:
    """Compute 11-dim feature vector for a node in a time window.

    Features:
      [0:3]  node_type one-hot (EOA, Contract, Token)
      [3]    tx_count in window
      [4]    total_value_in (log-scaled)
      [5]    total_value_out (log-scaled)
      [6]    unique_counterparties
      [7]    is_flash_loan_user
      [8]    is_known_dex
      [9]    is_known_lending
      [10]   profit_score
    """
    features = np.zeros(11, dtype=np.float32)

    addr = address.lower()

    # Node type (heuristic: contracts have code, but here we use event patterns)
    is_contract = addr in known_dex or addr in known_lending
    if is_contract:
        features[1] = 1.0  # Contract
    else:
        features[0] = 1.0  # EOA (default)

    total_in = 0.0
    total_out = 0.0
    counterparties: set[str] = set()
    has_flash_loan = False

    for e in events:
        if e.to_address == addr:
            total_in += float(e.amount)
            counterparties.add(e.from_address)
        if e.from_address == addr:
            total_out += float(e.amount)
            counterparties.add(e.to_address)
        if e.function_selector and e.function_selector.startswith("0x"):
            # Simple heuristic: flash loan selectors
            if e.function_selector[:10] in ("0xab9c4b5d", "0x5cffe9de"):
                has_flash_loan = True

    features[3] = len([e for e in events if e.from_address == addr or e.to_address == addr])
    features[4] = math.log1p(total_in)
    features[5] = math.log1p(total_out)
    features[6] = len(counterparties)
    features[7] = 1.0 if has_flash_loan else 0.0
    features[8] = 1.0 if addr in known_dex else 0.0
    features[9] = 1.0 if addr in known_lending else 0.0

    # Profit score
    if total_in > 0:
        features[10] = (total_out - total_in) / total_in
    else:
        features[10] = 0.0

    return features


def compute_edge_features(
    event,  # UnifiedEvent
    time2vec: Optional[Time2Vec] = None,
    selector_vocab: Optional[dict[str, int]] = None,
) -> np.ndarray:
    """Compute 30-dim feature vector for an edge.

    Features:
      [0:4]   edge_type one-hot (transfer, swap, call, liquidity)
      [4]     amount_normalized (log-scaled)
      [5]     price_impact
      [6:22]  time encoding (Time2Vec, 16-dim)
      [22:30] function_selector embedding (8-dim)
    """
    features = np.zeros(30, dtype=np.float32)

    # Edge type one-hot
    type_map = {"transfer": 0, "swap": 1, "call": 2, "liquidity": 3}
    idx = type_map.get(event.event_type, 0)
    features[idx] = 1.0

    # Amount (log-scaled)
    features[4] = math.log1p(float(event.amount))

    # Price impact
    features[5] = event.price_impact or 0.0

    # Time encoding
    if time2vec is not None:
        t = torch.tensor([float(event.timestamp)])
        with torch.no_grad():
            time_emb = time2vec(t).numpy().flatten()
        features[6:22] = time_emb[:16]
    else:
        # Fallback: simple sinusoidal encoding
        t = float(event.timestamp)
        for i in range(16):
            freq = 1.0 / (10000 ** (i / 16))
            if i % 2 == 0:
                features[6 + i] = math.sin(t * freq)
            else:
                features[6 + i] = math.cos(t * freq)

    # Function selector embedding (simple hash-based)
    if event.function_selector and selector_vocab:
        sel = event.function_selector[:10]
        if sel in selector_vocab:
            sel_idx = selector_vocab[sel]
            # Simple embedding: one-hot mod 8
            features[22 + (sel_idx % 8)] = 1.0

    return features
