#!/usr/bin/env python3
"""Script to train the MultiScaleDetector model.

Usage:
    python scripts/train.py --chain ethereum --epochs 100
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model.temporal_gnn import MultiScaleDetector
from model.train import train
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_graphs(graphs_path: Path) -> list:
    """Load pickled graph data and convert to model-compatible dicts."""
    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)

    model_data = []
    for g in graphs:
        if g.num_edges == 0:
            continue

        pyg_data = g.to_pyg_temporal_data()

        # Assemble node features matrix
        num_nodes = g.num_nodes
        node_feat = np.zeros((num_nodes, 11), dtype=np.float32)
        for idx, feat in g.node_features.items():
            node_feat[idx] = feat

        model_data.append({
            "src": pyg_data["src"],
            "dst": pyg_data["dst"],
            "t": pyg_data["t"],
            "msg": pyg_data["msg"],
            "node_features": torch.from_numpy(node_feat),
            "label": g.label or 0,
        })

    return model_data


def split_data(data: list, train_ratio: float, val_ratio: float):
    """Temporal split: earlier data for training, later for validation/test."""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def main():
    parser = argparse.ArgumentParser(description="Train MultiScaleDetector")
    parser.add_argument("--chain", default="ethereum")
    parser.add_argument("--graphs-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.chain)
    project_root = Path(__file__).resolve().parent.parent

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load data
    if args.graphs_path:
        graphs_path = Path(args.graphs_path)
    else:
        processed_dir = project_root / "data" / "processed"
        candidates = sorted(processed_dir.glob(f"graphs_{args.chain}_*.pkl"))
        if not candidates:
            logger.error("No graph files found. Run build_graphs.py first.")
            sys.exit(1)
        graphs_path = candidates[-1]

    logger.info(f"Loading graphs from {graphs_path}")
    data = load_graphs(graphs_path)
    logger.info(f"Loaded {len(data)} graph samples")

    if len(data) < 10:
        logger.error("Insufficient data for training.")
        sys.exit(1)

    # Split
    train_cfg = config.get("training", {})
    train_data, val_data, test_data = split_data(
        data,
        train_cfg.get("train_ratio", 0.7),
        train_cfg.get("val_ratio", 0.15),
    )
    logger.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Model
    model_cfg = config.get("model", {})
    model = MultiScaleDetector(
        node_feat_dim=config["graph"]["node_feature_dim"],
        edge_feat_dim=config["graph"]["edge_feature_dim"],
        embedding_dim=model_cfg.get("embedding_dim", 64),
        time_dim=config["graph"]["time2vec_dim"],
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.3),
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Train
    train_config = {
        "learning_rate": args.lr or train_cfg.get("learning_rate", 1e-4),
        "weight_decay": train_cfg.get("weight_decay", 1e-5),
        "epochs": args.epochs,
        "patience": train_cfg.get("patience", 10),
        "focal_loss_gamma": train_cfg.get("focal_loss_gamma", 2.0),
        "windows": config.get("graph", {}).get("windows", {}),
    }

    save_dir = project_root / "data" / "checkpoints"
    history = train(model, train_data, val_data, train_config, device, save_dir)

    logger.info("Training complete.")
    logger.info(f"Best val loss: {min(history['val_loss']):.4f}")
    if history["val_metrics"]:
        best_idx = np.argmin(history["val_loss"])
        best_metrics = history["val_metrics"][best_idx]
        logger.info(
            f"Best metrics: P={best_metrics['precision']:.3f} "
            f"R={best_metrics['recall']:.3f} F1={best_metrics['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
