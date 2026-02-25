#!/usr/bin/env python3
"""Script to collect blockchain data for attack detection training.

Usage:
    python scripts/collect_data.py --chain ethereum --from-block 14000000 --to-block 14100000
    python scripts/collect_data.py --chain bsc --address 0x...
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_collector.collector import AlchemyCollector, Web3EventCollector
from data_collector.storage import EventStorage
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect blockchain event data")
    parser.add_argument("--chain", default="ethereum", choices=["ethereum", "bsc"])
    parser.add_argument("--from-block", type=int, required=True)
    parser.add_argument("--to-block", type=int, required=True)
    parser.add_argument("--address", type=str, default=None, help="Filter by address")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    config = load_config(args.chain)
    api_key = config["chain"].get("alchemy_api_key", "")
    if not api_key:
        logger.error("ALCHEMY_API_KEY not set. Export it as environment variable.")
        sys.exit(1)

    db_path = args.db_path or str(
        Path(__file__).resolve().parent.parent / "data" / f"events_{args.chain}.db"
    )
    storage = EventStorage(db_path)
    collector = AlchemyCollector(api_key, chain=args.chain)

    logger.info(f"Collecting data: chain={args.chain} blocks={args.from_block}-{args.to_block}")

    total_events = 0
    from_block = args.from_block

    while from_block < args.to_block:
        batch_end = min(from_block + args.batch_size, args.to_block)
        logger.info(f"Fetching blocks {from_block}-{batch_end}...")

        events = list(collector.get_asset_transfers(
            from_block=from_block,
            to_block=batch_end,
            from_address=args.address,
        ))

        if events:
            stored = storage.store_events(events)
            total_events += stored
            logger.info(f"  Stored {stored} events (total: {total_events})")

        from_block = batch_end + 1

    logger.info(f"Collection complete. Total events: {total_events}")
    storage.close()


if __name__ == "__main__":
    main()
