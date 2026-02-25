"""Post-detection attack analysis and explanation module."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..clustering.address_clustering import AddressCluster
from ..data_collector.events import UnifiedEvent

logger = logging.getLogger(__name__)


@dataclass
class AttackAnalysis:
    """Analysis result for a detected attack."""

    # Identification
    detection_window: tuple[int, int]  # (start_block, end_block)
    confidence: float  # P(attack) from model
    attack_class: str  # predicted attack type

    # Participants
    attacker_cluster: set[str]  # related attacker addresses
    victim_contracts: list[str]
    victim_tokens: list[str]

    # Timeline
    preparation_events: list[UnifiedEvent]  # setup phase
    manipulation_events: list[UnifiedEvent]  # price manipulation phase
    profit_events: list[UnifiedEvent]  # profit extraction phase

    # Financial
    estimated_profit_eth: float
    estimated_profit_usd: Optional[float]

    def summary(self) -> str:
        lines = [
            f"=== Attack Detection ===",
            f"Window: blocks {self.detection_window[0]}-{self.detection_window[1]}",
            f"Confidence: {self.confidence:.2%}",
            f"Type: {self.attack_class}",
            f"Attacker addresses: {len(self.attacker_cluster)}",
            f"Victim contracts: {', '.join(self.victim_contracts[:3])}",
            f"Estimated profit: {self.estimated_profit_eth:.4f} ETH",
        ]
        if self.estimated_profit_usd:
            lines.append(f"Estimated profit (USD): ${self.estimated_profit_usd:,.2f}")
        lines.append(f"Timeline: {len(self.preparation_events)} prep → "
                     f"{len(self.manipulation_events)} manipulation → "
                     f"{len(self.profit_events)} profit events")
        return "\n".join(lines)


class AttackAnalyzer:
    """Analyze detected attack windows to produce human-readable explanations."""

    # Attack type classification based on temporal patterns
    ATTACK_PATTERNS = {
        "slow_manipulation": {
            "min_duration_blocks": 100,
            "has_repeated_swaps": True,
        },
        "governance": {
            "min_duration_blocks": 5000,
            "has_governance_calls": True,
        },
        "oracle": {
            "has_multi_pool_swaps": True,
        },
        "slow_rug": {
            "min_duration_blocks": 10000,
            "has_liquidity_removal": True,
        },
        "sandwich": {
            "max_duration_blocks": 5,
        },
    }

    def __init__(self, known_dex: set[str], known_lending: set[str]):
        self.known_dex = known_dex
        self.known_lending = known_lending

    def analyze(
        self,
        events: list[UnifiedEvent],
        window: tuple[int, int],
        confidence: float,
        address_cluster: Optional[AddressCluster] = None,
    ) -> AttackAnalysis:
        """Analyze events in a detected attack window."""
        # Identify suspicious addresses (high profit)
        address_profit = self._compute_address_profits(events)
        suspicious = {addr for addr, profit in address_profit.items() if profit > 0}

        # Expand via clustering
        attacker_cluster = set()
        if address_cluster and suspicious:
            for addr in suspicious:
                root = address_cluster.find(addr)
                clusters = address_cluster.get_clusters()
                if root in clusters:
                    attacker_cluster.update(clusters[root])
        if not attacker_cluster:
            attacker_cluster = suspicious

        # Identify victims
        victim_contracts = []
        victim_tokens = set()
        for e in events:
            if e.contract_address in self.known_dex or e.contract_address in self.known_lending:
                if e.contract_address not in victim_contracts:
                    victim_contracts.append(e.contract_address)
            if e.token_address:
                victim_tokens.add(e.token_address)

        # Split timeline into phases
        prep, manip, profit = self._split_phases(events, attacker_cluster)

        # Classify attack type
        attack_class = self._classify_attack(events, window)

        # Estimate profit
        total_profit = sum(
            max(p, 0) for addr, p in address_profit.items()
            if addr in attacker_cluster
        )

        return AttackAnalysis(
            detection_window=window,
            confidence=confidence,
            attack_class=attack_class,
            attacker_cluster=attacker_cluster,
            victim_contracts=victim_contracts,
            victim_tokens=list(victim_tokens),
            preparation_events=prep,
            manipulation_events=manip,
            profit_events=profit,
            estimated_profit_eth=total_profit,
            estimated_profit_usd=None,
        )

    def _compute_address_profits(
        self, events: list[UnifiedEvent]
    ) -> dict[str, float]:
        """Compute net profit for each address."""
        balance: dict[str, float] = defaultdict(float)
        for e in events:
            amt = float(e.amount)
            balance[e.to_address] += amt
            balance[e.from_address] -= amt
        return dict(balance)

    def _split_phases(
        self,
        events: list[UnifiedEvent],
        attacker_addrs: set[str],
    ) -> tuple[list, list, list]:
        """Split events into preparation, manipulation, and profit phases."""
        if not events:
            return [], [], []

        total = len(events)
        # Simple heuristic: first 20% = prep, middle 60% = manipulation, last 20% = profit
        prep_end = total // 5
        profit_start = total - total // 5

        prep = events[:prep_end]
        manip = events[prep_end:profit_start]
        profit = events[profit_start:]

        return prep, manip, profit

    def _classify_attack(
        self,
        events: list[UnifiedEvent],
        window: tuple[int, int],
    ) -> str:
        """Classify attack type based on temporal and behavioral patterns."""
        duration = window[1] - window[0]

        swap_count = sum(1 for e in events if e.event_type == "swap")
        liquidity_count = sum(1 for e in events if e.event_type == "liquidity")
        unique_pools = len(set(e.contract_address for e in events if e.event_type == "swap"))

        if duration <= 5:
            return "sandwich"
        if duration >= 10000 and liquidity_count > 0:
            return "slow_rug"
        if duration >= 5000:
            return "governance"
        if unique_pools > 2 and swap_count > 5:
            return "oracle"
        if swap_count > 3 and duration >= 100:
            return "slow_manipulation"

        return "unknown"
