"""Evaluation utilities for token-level NER models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import build_classification_report, flatten_predictions_and_labels


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_pad_id: int = -100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect flattened predictions and labels with padding removed."""
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            logits = model(input_ids, lengths)
            y_true, y_pred = flatten_predictions_and_labels(
                logits,
                labels,
                ignore_index=label_pad_id,
            )
            all_true.append(y_true)
            all_pred.append(y_pred)

    return np.concatenate(all_true), np.concatenate(all_pred)


def save_classification_report(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: List[str],
    output_path: str | Path,
    label_pad_id: int = -100,
) -> str:
    """Create and save sklearn classification_report."""
    y_true, y_pred = collect_predictions(
        model,
        dataloader,
        device,
        label_pad_id=label_pad_id,
    )
    report = build_classification_report(y_true, y_pred, label_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report
