"""Metrics for token-level sequence labeling with padded-label masking."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute accuracy while ignoring padded label positions."""
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(ignore_index)
    total = mask.sum().item()
    if total == 0:
        return 0.0
    correct = predictions.eq(labels).masked_select(mask).sum().item()
    return correct / total


def flatten_predictions_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten predictions and labels, removing padding positions."""
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(ignore_index)
    y_pred = predictions.masked_select(mask).detach().cpu().numpy()
    y_true = labels.masked_select(mask).detach().cpu().numpy()
    return y_true, y_pred


def build_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    label_names: List[str],
) -> str:
    """Create sklearn classification_report with stable label ordering."""
    labels = list(range(len(label_names)))
    return classification_report(
        list(y_true),
        list(y_pred),
        labels=labels,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
