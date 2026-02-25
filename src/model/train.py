"""Training loop for MultiScaleDetector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import FocalLoss
from .temporal_gnn import MultiScaleDetector

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def prepare_multi_scale_inputs(
    graph_data: dict,
    window_configs: dict,
    device: torch.device,
) -> dict:
    """Prepare inputs for MultiScaleDetector from a graph data dict.

    Args:
        graph_data: dict with keys from TemporalGraphData.to_pyg_temporal_data()
                    plus 'node_features' array.
        window_configs: dict with short/medium/long window sizes.
        device: target device.

    Returns:
        dict of tensors for model forward().
    """
    src = graph_data["src"].to(device)
    dst = graph_data["dst"].to(device)
    t = graph_data["t"].to(device)
    msg = graph_data["msg"].to(device)
    node_feat = graph_data["node_features"].to(device)

    num_edges = len(src)

    # Split edges into windows by proportion
    # Short: first 10%, Medium: first 50%, Long: all
    short_end = max(1, num_edges // 10)
    medium_end = max(1, num_edges // 2)

    return {
        "short_node_features": node_feat,
        "short_sources": src[:short_end],
        "short_destinations": dst[:short_end],
        "short_timestamps": t[:short_end],
        "short_edge_features": msg[:short_end],
        "medium_timestamps": t[:medium_end],
        "medium_edge_features": msg[:medium_end],
        "long_timestamps": t,
        "long_edge_features": msg,
    }


def train_epoch(
    model: MultiScaleDetector,
    train_data: list[dict],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    window_configs: dict,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for graph_data in train_data:
        optimizer.zero_grad()
        inputs = prepare_multi_scale_inputs(graph_data, window_configs, device)
        logits = model(**inputs)

        label = torch.tensor([graph_data["label"]], device=device)
        loss = loss_fn(logits.unsqueeze(0), label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(train_data), 1)


@torch.no_grad()
def evaluate(
    model: MultiScaleDetector,
    eval_data: list[dict],
    loss_fn: nn.Module,
    device: torch.device,
    window_configs: dict,
) -> dict:
    """Evaluate model. Returns dict with loss, predictions, and labels."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for graph_data in eval_data:
        inputs = prepare_multi_scale_inputs(graph_data, window_configs, device)
        logits = model(**inputs)

        label = torch.tensor([graph_data["label"]], device=device)
        loss = loss_fn(logits.unsqueeze(0), label)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax().item()

        all_preds.append(pred)
        all_labels.append(graph_data["label"])
        all_probs.append(probs[1].item())  # P(attack)

    return {
        "loss": total_loss / max(len(eval_data), 1),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def train(
    model: MultiScaleDetector,
    train_data: list[dict],
    val_data: list[dict],
    config: dict,
    device: torch.device,
    save_dir: Optional[Path] = None,
) -> dict:
    """Full training loop with early stopping.

    Args:
        model: MultiScaleDetector instance.
        train_data: list of graph data dicts (with 'label' key).
        val_data: list of graph data dicts for validation.
        config: training config dict.
        device: target device.
        save_dir: directory to save best model checkpoint.

    Returns:
        dict with training history.
    """
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get("epochs", 100))

    loss_fn = FocalLoss(gamma=config.get("focal_loss_gamma", 2.0))
    early_stopping = EarlyStopping(patience=config.get("patience", 10))

    window_configs = config.get("windows", {})
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_val_loss = float("inf")

    for epoch in range(config.get("epochs", 100)):
        train_loss = train_epoch(
            model, train_data, optimizer, loss_fn, device, window_configs,
        )
        val_result = evaluate(model, val_data, loss_fn, device, window_configs)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_result["loss"])

        # Compute metrics
        preds = np.array(val_result["predictions"])
        labels = np.array(val_result["labels"])
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        metrics = {"precision": precision, "recall": recall, "f1": f1}
        history["val_metrics"].append(metrics)

        logger.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

        # Save best model
        if val_result["loss"] < best_val_loss and save_dir:
            best_val_loss = val_result["loss"]
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            logger.info(f"Saved best model (val_loss={best_val_loss:.4f})")

        if early_stopping.step(val_result["loss"]):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    return history
