"""Known attack dataset and labeling utilities.

Ground truth sources:
  - DeFiHackLabs (550+ incidents)
  - DeFiScope D1 (95 attacks)
  - DeFort D2 (54 attacks)
  - Manual curation from rekt.news
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AttackRecord:
    """A known price manipulation attack record."""

    name: str
    chain: str  # "ethereum" | "bsc"
    attack_type: str  # slow_manipulation | governance | oracle | slow_rug | sandwich
    tx_hashes: list[str]  # known attack transaction hashes
    attacker_addresses: list[str]
    victim_protocol: str
    victim_contract: Optional[str]
    start_block: int
    end_block: int
    loss_usd: Optional[float]
    source: str  # "defihacklabs" | "defiscope" | "defort" | "manual"
    notes: str = ""


# Curated list of cross-transaction attacks for initial dataset
KNOWN_CROSS_TX_ATTACKS: list[dict] = [
    {
        "name": "INUKO",
        "chain": "bsc",
        "attack_type": "slow_manipulation",
        "tx_hashes": [],  # To be filled from DeFiHackLabs
        "attacker_addresses": [],
        "victim_protocol": "INUKO Token",
        "victim_contract": "0xea51801b8f5b88543ddad3d1727400c15b209d8f",
        "start_block": 0,  # ~48h before exploit
        "end_block": 0,
        "loss_usd": None,
        "source": "defiscope",
        "notes": "48-hour, ~57000 block cross-tx manipulation",
    },
    {
        "name": "Beanstalk",
        "chain": "ethereum",
        "attack_type": "governance",
        "tx_hashes": [
            "0xcd314668aaa9bbfebaf1a0bd2b6553d01dd58899c508d4729fa7311dc5d33ad7",
        ],
        "attacker_addresses": [
            "0x1c5dcdd006ea78a7e4783f9e6021c32935a10fb4",
        ],
        "victim_protocol": "Beanstalk",
        "victim_contract": "0xC1E088fC1323b20BCBee9bd1B9fC9546db5624C5",
        "start_block": 14595905,  # BIP-18 proposal submission
        "end_block": 14602790,   # Exploit execution
        "loss_usd": 182_000_000,
        "source": "manual",
        "notes": "Governance attack: BIP-18 proposal + flash loan vote + drain",
    },
    {
        "name": "Mango Markets",
        "chain": "ethereum",  # Solana actually, but listed for reference
        "attack_type": "oracle",
        "tx_hashes": [],
        "attacker_addresses": [],
        "victim_protocol": "Mango Markets",
        "victim_contract": None,
        "start_block": 0,
        "end_block": 0,
        "loss_usd": 117_000_000,
        "source": "manual",
        "notes": "Multi-step cross-exchange oracle manipulation (Solana)",
    },
]


def load_attack_records(labels_dir: Path) -> list[AttackRecord]:
    """Load attack records from JSON files in labels directory."""
    records: list[AttackRecord] = []

    for json_file in sorted(labels_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    records.append(AttackRecord(**item))
            else:
                records.append(AttackRecord(**data))
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    logger.info(f"Loaded {len(records)} attack records from {labels_dir}")
    return records


def save_attack_records(records: list[AttackRecord], output_path: Path) -> None:
    """Save attack records to JSON."""
    data = []
    for r in records:
        d = {
            "name": r.name,
            "chain": r.chain,
            "attack_type": r.attack_type,
            "tx_hashes": r.tx_hashes,
            "attacker_addresses": r.attacker_addresses,
            "victim_protocol": r.victim_protocol,
            "victim_contract": r.victim_contract,
            "start_block": r.start_block,
            "end_block": r.end_block,
            "loss_usd": r.loss_usd,
            "source": r.source,
            "notes": r.notes,
        }
        data.append(d)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(records)} attack records to {output_path}")


def generate_window_labels(
    attacks: list[AttackRecord],
    window_size: int,
    stride: int,
    min_block: int,
    max_block: int,
) -> dict[tuple[int, int], int]:
    """Generate binary labels for sliding windows.

    A window is labeled as attack (1) if it overlaps with any known
    attack's block range. Otherwise labeled as normal (0).

    Returns:
        dict mapping (start_block, end_block) -> label (0 or 1)
    """
    labels: dict[tuple[int, int], int] = {}
    start = min_block

    while start + window_size <= max_block + 1:
        end = start + window_size
        is_attack = False

        for attack in attacks:
            if attack.start_block == 0 or attack.end_block == 0:
                continue
            # Check overlap
            if start < attack.end_block and end > attack.start_block:
                is_attack = True
                break

        labels[(start, end)] = 1 if is_attack else 0
        start += stride

    num_attack = sum(v for v in labels.values())
    num_normal = len(labels) - num_attack
    logger.info(
        f"Generated {len(labels)} window labels: "
        f"{num_attack} attack, {num_normal} normal"
    )
    return labels
