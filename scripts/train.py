"""Training loop for NER encoder models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import masked_accuracy


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    label_pad_id: int = -100,
) -> Tuple[float, float]:
    """Run one train or evaluation epoch.

    If optimizer is supplied, the model is trained; otherwise it is evaluated.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        lengths = batch["lengths"]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(input_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        mask = labels.ne(label_pad_id)
        num_tokens = mask.sum().item()
        if num_tokens > 0:
            total_loss += loss.item() * num_tokens
            predictions = logits.argmax(dim=-1)
            total_correct += predictions.eq(labels).masked_select(mask).sum().item()
            total_tokens += num_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return mean_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    label_pad_id: int,
    save_path: str | Path,
) -> Dict[str, list]:
    """Train a model and save the best checkpoint by validation accuracy."""
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -1.0
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            label_pad_id=label_pad_id,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            label_pad_id=label_pad_id,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train loss={train_loss:.4f}, train acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, val acc={val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), save_path)

    return history
