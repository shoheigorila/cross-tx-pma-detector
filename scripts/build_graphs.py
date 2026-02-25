#!/usr/bin/env python3
"""Script to build temporal graphs from collected event data.

Usage:
    python scripts/build_graphs.py --chain ethereum --window-size 1000 --stride 200
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_collector.labels import load_attack_records, generate_window_labels
from data_collector.storage import EventStorage
from graph_builder.builder import TemporalGraphBuilder
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build temporal graphs from events")
    parser.add_argument("--chain", default="ethereum", choices=["ethereum", "bsc"])
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.chain)
    project_root = Path(__file__).resolve().parent.parent

    db_path = args.db_path or str(project_root / "data" / f"events_{args.chain}.db")
    output_dir = Path(args.output_dir or str(project_root / "data" / "processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load known contracts
    known_contracts = config.get("known_contracts", {})
    known_dex = set()
    known_lending = set()
    for addr in known_contracts.get("dex", {}).values():
        known_dex.add(addr.lower())
    for addr in known_contracts.get("lending", {}).values():
        known_lending.add(addr.lower())

    # Load events
    storage = EventStorage(db_path)
    min_block, max_block = storage.get_block_range()
    logger.info(f"Event DB: {storage.count()} events, blocks {min_block}-{max_block}")

    if storage.count() == 0:
        logger.error("No events in database. Run collect_data.py first.")
        sys.exit(1)

    events = storage.query_events(min_block, max_block)
    logger.info(f"Loaded {len(events)} events")

    # Load attack labels
    labels_dir = project_root / "data" / "labels"
    attacks = []
    if labels_dir.exists():
        attacks = load_attack_records(labels_dir)
    window_labels = generate_window_labels(
        attacks, args.window_size, args.stride, min_block, max_block,
    )

    # Build graphs
    builder = TemporalGraphBuilder(
        known_dex=known_dex,
        known_lending=known_lending,
    )
    graphs = builder.build_sliding_windows(
        events, args.window_size, args.stride, labels=window_labels,
    )

    # Save
    output_file = output_dir / f"graphs_{args.chain}_w{args.window_size}_s{args.stride}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(graphs, f)
    logger.info(f"Saved {len(graphs)} graphs to {output_file}")

    # Stats
    attack_count = sum(1 for g in graphs if g.label == 1)
    normal_count = sum(1 for g in graphs if g.label == 0)
    logger.info(f"Attack windows: {attack_count}, Normal windows: {normal_count}")

    storage.close()


if __name__ == "__main__":
    main()
