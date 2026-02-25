#!/usr/bin/env python3
"""Script to evaluate trained model on test data.

Usage:
    python scripts/evaluate.py --chain ethereum --checkpoint data/checkpoints/best_model.pt
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model.evaluate import compute_metrics, per_class_metrics, print_evaluation_report
from model.losses import FocalLoss
from model.temporal_gnn import MultiScaleDetector
from model.train import evaluate, prepare_multi_scale_inputs
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--chain", default="ethereum")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--graphs-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.chain)
    project_root = Path(__file__).resolve().parent.parent

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model_cfg = config.get("model", {})
    model = MultiScaleDetector(
        node_feat_dim=config["graph"]["node_feature_dim"],
        edge_feat_dim=config["graph"]["edge_feature_dim"],
        embedding_dim=model_cfg.get("embedding_dim", 64),
        time_dim=config["graph"]["time2vec_dim"],
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.3),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    logger.info(f"Loaded model from {args.checkpoint}")

    # Load test data (last 15% of temporal split)
    if args.graphs_path:
        graphs_path = Path(args.graphs_path)
    else:
        processed_dir = project_root / "data" / "processed"
        candidates = sorted(processed_dir.glob(f"graphs_{args.chain}_*.pkl"))
        if not candidates:
            logger.error("No graph files found.")
            sys.exit(1)
        graphs_path = candidates[-1]

    from scripts.train import load_graphs  # noqa
    data = load_graphs(graphs_path)

    # Take test split (last 15%)
    train_ratio = config.get("training", {}).get("train_ratio", 0.7)
    val_ratio = config.get("training", {}).get("val_ratio", 0.15)
    test_start = int(len(data) * (train_ratio + val_ratio))
    test_data = data[test_start:]
    logger.info(f"Test set: {len(test_data)} samples")

    if len(test_data) == 0:
        logger.error("No test data.")
        sys.exit(1)

    # Evaluate
    loss_fn = FocalLoss(gamma=config.get("training", {}).get("focal_loss_gamma", 2.0))
    window_configs = config.get("graph", {}).get("windows", {})

    result = evaluate(model, test_data, loss_fn, device, window_configs)

    # Compute metrics
    metrics = compute_metrics(
        result["labels"],
        result["predictions"],
        result["probabilities"],
    )

    attack_classes = [
        "normal",
        "slow_manipulation",
        "governance",
        "oracle",
        "slow_rug",
        "sandwich",
    ]
    per_class = per_class_metrics(result["labels"], result["predictions"])

    print_evaluation_report(metrics, per_class)


if __name__ == "__main__":
    main()
