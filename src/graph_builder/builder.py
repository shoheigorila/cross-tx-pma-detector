"""Temporal graph construction from unified events."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from ..data_collector.events import UnifiedEvent
from .features import Time2Vec, compute_edge_features, compute_node_features

logger = logging.getLogger(__name__)


class TemporalGraphData:
    """Container for a Continuous-Time Dynamic Graph (CTDG).

    Stores events as sorted edge lists with node/edge features,
    compatible with PyG TemporalData and TGN-style models.
    """

    def __init__(self):
        self.sources: list[int] = []       # source node indices
        self.destinations: list[int] = []  # destination node indices
        self.timestamps: list[float] = []  # event timestamps
        self.edge_features: list[np.ndarray] = []
        self.node_features: dict[int, np.ndarray] = {}

        # Mappings
        self.address_to_idx: dict[str, int] = {}
        self.idx_to_address: dict[int, str] = {}
        self._next_idx = 0

        # Metadata
        self.label: Optional[int] = None  # 0=normal, 1=attack
        self.block_range: tuple[int, int] = (0, 0)

    def _get_or_create_node(self, address: str) -> int:
        addr = address.lower()
        if addr not in self.address_to_idx:
            self.address_to_idx[addr] = self._next_idx
            self.idx_to_address[self._next_idx] = addr
            self._next_idx += 1
        return self.address_to_idx[addr]

    @property
    def num_nodes(self) -> int:
        return self._next_idx

    @property
    def num_edges(self) -> int:
        return len(self.sources)

    def to_pyg_temporal_data(self) -> dict:
        """Convert to dict compatible with PyG TemporalData."""
        return {
            "src": torch.tensor(self.sources, dtype=torch.long),
            "dst": torch.tensor(self.destinations, dtype=torch.long),
            "t": torch.tensor(self.timestamps, dtype=torch.float),
            "msg": torch.stack([torch.from_numpy(f) for f in self.edge_features])
            if self.edge_features
            else torch.zeros(0, 30),
            "num_nodes": self.num_nodes,
        }


class TemporalGraphBuilder:
    """Build CTDG from a stream of UnifiedEvents."""

    def __init__(
        self,
        known_dex: Optional[set[str]] = None,
        known_lending: Optional[set[str]] = None,
        time2vec_dim: int = 16,
    ):
        self.known_dex = known_dex or set()
        self.known_lending = known_lending or set()
        self.time2vec = Time2Vec(out_dim=time2vec_dim)
        self.selector_vocab: dict[str, int] = {}
        self._selector_counter = 0

    def _register_selector(self, selector: Optional[str]) -> None:
        if selector and selector[:10] not in self.selector_vocab:
            self.selector_vocab[selector[:10]] = self._selector_counter
            self._selector_counter += 1

    def build(
        self,
        events: list[UnifiedEvent],
        label: Optional[int] = None,
    ) -> TemporalGraphData:
        """Build a temporal graph from a list of events (assumed sorted by block/time)."""
        graph = TemporalGraphData()

        if not events:
            return graph

        graph.block_range = (events[0].block_number, events[-1].block_number)
        graph.label = label

        # Build edges
        for event in events:
            src_idx = graph._get_or_create_node(event.from_address)
            dst_idx = graph._get_or_create_node(event.to_address)

            graph.sources.append(src_idx)
            graph.destinations.append(dst_idx)
            graph.timestamps.append(float(event.timestamp))

            self._register_selector(event.function_selector)
            edge_feat = compute_edge_features(
                event,
                time2vec=self.time2vec,
                selector_vocab=self.selector_vocab,
            )
            graph.edge_features.append(edge_feat)

        # Build node features
        address_events: dict[str, list] = defaultdict(list)
        for event in events:
            address_events[event.from_address].append(event)
            address_events[event.to_address].append(event)

        for addr, idx in graph.address_to_idx.items():
            node_feat = compute_node_features(
                addr,
                address_events.get(addr, []),
                self.known_dex,
                self.known_lending,
            )
            graph.node_features[idx] = node_feat

        logger.info(
            f"Built graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
            f"blocks {graph.block_range[0]}-{graph.block_range[1]}"
        )
        return graph

    def build_sliding_windows(
        self,
        events: list[UnifiedEvent],
        window_size: int,
        stride: int,
        labels: Optional[dict[tuple[int, int], int]] = None,
    ) -> list[TemporalGraphData]:
        """Build multiple graphs using sliding window over block numbers."""
        if not events:
            return []

        min_block = events[0].block_number
        max_block = events[-1].block_number

        # Group events by block for efficient slicing
        block_events: dict[int, list[UnifiedEvent]] = defaultdict(list)
        for e in events:
            block_events[e.block_number].append(e)

        graphs: list[TemporalGraphData] = []
        start = min_block

        while start + window_size <= max_block + 1:
            end = start + window_size
            window_events = []
            for b in range(start, end):
                if b in block_events:
                    window_events.extend(block_events[b])

            if window_events:
                label = None
                if labels:
                    label = labels.get((start, end), 0)
                g = self.build(window_events, label=label)
                graphs.append(g)

            start += stride

        logger.info(f"Built {len(graphs)} windows (size={window_size}, stride={stride})")
        return graphs
