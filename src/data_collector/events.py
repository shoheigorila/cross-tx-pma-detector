"""Unified event data model for cross-transaction analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class UnifiedEvent:
    """A normalised blockchain event that unifies transfers, swaps, calls,
    and liquidity operations into a single schema."""

    tx_hash: str
    block_number: int
    timestamp: int
    event_type: str  # "transfer" | "swap" | "call" | "liquidity"
    from_address: str
    to_address: str
    token_address: str  # ETH represented as 0xEeee...
    amount: Decimal

    # Swap-specific fields
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    amount_in: Optional[Decimal] = None
    amount_out: Optional[Decimal] = None

    # Metadata
    function_selector: Optional[str] = None
    gas_price: int = 0
    contract_address: str = ""
    log_index: int = 0

    # Derived (set after construction via object.__setattr__ if needed)
    price_impact: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "token_address": self.token_address,
            "amount": str(self.amount),
            "token_in": self.token_in,
            "token_out": self.token_out,
            "amount_in": str(self.amount_in) if self.amount_in else None,
            "amount_out": str(self.amount_out) if self.amount_out else None,
            "function_selector": self.function_selector,
            "gas_price": self.gas_price,
            "contract_address": self.contract_address,
            "log_index": self.log_index,
            "price_impact": self.price_impact,
        }
