"""Address clustering heuristics for linking related attacker addresses."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from ..data_collector.events import UnifiedEvent

logger = logging.getLogger(__name__)

ETH_ADDRESS = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE".lower()


class AddressCluster:
    """Union-Find based address clustering."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, addr: str) -> str:
        addr = addr.lower()
        if addr not in self.parent:
            self.parent[addr] = addr
        if self.parent[addr] != addr:
            self.parent[addr] = self.find(self.parent[addr])
        return self.parent[addr]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a.lower())
        rb = self.find(b.lower())
        if ra != rb:
            self.parent[ra] = rb

    def get_clusters(self) -> dict[str, set[str]]:
        """Return mapping from cluster root -> set of addresses."""
        clusters: dict[str, set[str]] = defaultdict(set)
        for addr in self.parent:
            root = self.find(addr)
            clusters[root].add(addr)
        return dict(clusters)


class GasFundingClusterer:
    """Cluster addresses that receive initial gas funding from the same source.

    Heuristic: If address A sends ETH to addresses B and C within a short
    time window, and B/C had no prior ETH balance, they are likely controlled
    by the same entity.
    """

    def __init__(self, time_window_blocks: int = 100, max_hops: int = 2):
        self.time_window = time_window_blocks
        self.max_hops = max_hops

    def cluster(self, events: list[UnifiedEvent]) -> AddressCluster:
        ac = AddressCluster()

        # Find ETH transfers (gas funding)
        eth_sends: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for e in events:
            if e.token_address == ETH_ADDRESS and e.event_type == "transfer":
                eth_sends[e.from_address].append((e.to_address, e.block_number))

        # If sender funds multiple addresses within the time window, link them
        for sender, recipients in eth_sends.items():
            recipients.sort(key=lambda x: x[1])
            for i in range(len(recipients)):
                for j in range(i + 1, len(recipients)):
                    addr_i, block_i = recipients[i]
                    addr_j, block_j = recipients[j]
                    if block_j - block_i <= self.time_window:
                        ac.union(addr_i, addr_j)
                    else:
                        break

        return ac


class TemporalCooccurrenceClusterer:
    """Cluster addresses that frequently appear together in the same blocks.

    Heuristic: Addresses that transact in the same blocks across multiple
    occurrences are likely coordinated (e.g., attack bot wallets).
    """

    def __init__(self, block_threshold: int = 10, min_cooccurrences: int = 3):
        self.block_threshold = block_threshold
        self.min_cooccurrences = min_cooccurrences

    def cluster(self, events: list[UnifiedEvent]) -> AddressCluster:
        ac = AddressCluster()

        # Map blocks -> active addresses
        block_addresses: dict[int, set[str]] = defaultdict(set)
        for e in events:
            block_addresses[e.block_number].add(e.from_address)
            block_addresses[e.block_number].add(e.to_address)

        # Count pairwise co-occurrences within close blocks
        cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
        sorted_blocks = sorted(block_addresses.keys())

        for i, block in enumerate(sorted_blocks):
            # Look at addresses in nearby blocks
            nearby_addrs: set[str] = set()
            for j in range(max(0, i - self.block_threshold), min(len(sorted_blocks), i + self.block_threshold + 1)):
                nearby_addrs.update(block_addresses[sorted_blocks[j]])

            addrs = sorted(block_addresses[block])
            for a in addrs:
                for b in nearby_addrs:
                    if a < b:
                        cooccurrence[(a, b)] += 1

        # Link addresses exceeding threshold
        for (a, b), count in cooccurrence.items():
            if count >= self.min_cooccurrences:
                ac.union(a, b)

        return ac


class ContractCreationClusterer:
    """Cluster addresses based on contract creation patterns.

    Heuristic: Contracts deployed by the same EOA are linked.
    """

    def cluster(self, events: list[UnifiedEvent]) -> AddressCluster:
        ac = AddressCluster()

        # Find contract creation events (to_address = 0x0 or specific pattern)
        for e in events:
            if e.event_type == "call" and e.to_address == "0x" + "0" * 40:
                # Link deployer with the new contract
                if e.contract_address:
                    ac.union(e.from_address, e.contract_address)

        return ac


def merge_clusters(*clusters: AddressCluster) -> AddressCluster:
    """Merge multiple AddressCluster instances into one."""
    merged = AddressCluster()
    for cluster in clusters:
        for addr in cluster.parent:
            root = cluster.find(addr)
            merged.union(addr, root)
    return merged
