"""Main script to run dataset analysis, training, evaluation, and plotting."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

# Allow running either `python scripts/main.py` or `python -m scripts.main`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import (  # noqa: E402
    SpecialTokens,
    build_vocab_from_train_split,
    describe_dataset,
    get_label_names,
    load_conll_dataset,
    make_dataloaders,
)
from models.model import EncoderTagger  # noqa: E402
from scripts.evaluate import save_classification_report  # noqa: E402
from scripts.train import train_model  # noqa: E402
from utils.visualization import plot_history  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_batch_shape(name: str, dataloader) -> None:
    batch = next(iter(dataloader))
    print(f"{name} first batch input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"{name} first batch labels shape:    {tuple(batch['labels'].shape)}")
    print(f"{name} first batch lengths shape:   {tuple(batch['lengths'].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CoNLL-2003 NER sequence labeling")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for num_epochs")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    vocab_cfg = config["vocab"]
    model_cfg = config["model"]
    outputs_cfg = config["outputs"]

    for directory in [
        outputs_cfg["root_dir"],
        outputs_cfg["plot_dir"],
        outputs_cfg["report_dir"],
        outputs_cfg["saved_model_dir"],
    ]:
        Path(PROJECT_ROOT / directory).mkdir(parents=True, exist_ok=True)

    device = resolve_device(train_cfg["device"])
    print(f"Using device: {device}")

    print("\n=== Part 1(a): Loading and describing dataset ===")
    dataset_dict = load_conll_dataset(
        dataset_cfg["name"],
        revision=dataset_cfg.get("revision", "convert/parquet"),
    )
    print(describe_dataset(dataset_dict))

    print("\n=== Part 1(c-d): Building vocabulary from train split ===")
    special_tokens = SpecialTokens(**vocab_cfg["special_tokens"])
    vocab = build_vocab_from_train_split(
        dataset_dict["train"],
        token_column=dataset_cfg["token_column"],
        cutoff=vocab_cfg["cutoff"],
        special_tokens=special_tokens,
    )
    print(f"Vocabulary size: {vocab.vocab_size()}")
    print(f"Token with id=0: {vocab.id_to_token(0)}")
    print(f"ID of token 'quick': {vocab.token_to_id('quick')}")
    if vocab.token_to_id("quick") == vocab.unk_id:
        print("Note: 'quick' is not in the training vocabulary, so it maps to <unk>.")

    print("\n=== Part 2: Dataloaders with collate_fn ===")
    dataloaders = make_dataloaders(
        dataset_dict,
        vocab=vocab,
        batch_size=train_cfg["batch_size"],
        token_column=dataset_cfg["token_column"],
        label_column=dataset_cfg["label_column"],
        label_pad_id=train_cfg["label_pad_id"],
    )
    print_batch_shape("Train", dataloaders["train"])
    print_batch_shape("Validation", dataloaders["validation"])

    label_names = get_label_names(dataset_dict, dataset_cfg["label_column"])
    num_labels = len(label_names)
    num_epochs = args.epochs if args.epochs is not None else train_cfg["num_epochs"]

    print("\n=== Part 3: Training encoder variants ===")
    summary_rows = []
    for rnn_type in config["experiments"]:
        title = f"Bi{rnn_type.upper()}" if rnn_type != "rnn" else "Bidirectional RNN"
        print(f"\n--- Training {title} ---")
        model = EncoderTagger(
            vocab_size=vocab.vocab_size(),
            num_labels=num_labels,
            embedding_size=model_cfg["embedding_size"],
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            bidirectional=model_cfg["bidirectional"],
            rnn_type=rnn_type,
            pad_token_id=vocab.pad_id,
        ).to(device)

        save_path = PROJECT_ROOT / outputs_cfg["saved_model_dir"] / f"{rnn_type}_best.pt"
        history = train_model(
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["validation"],
            device=device,
            num_epochs=num_epochs,
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            label_pad_id=train_cfg["label_pad_id"],
            save_path=save_path,
        )

        # Evaluate the best checkpoint on validation and test sets.
        model.load_state_dict(torch.load(save_path, map_location=device))

        plot_path = PROJECT_ROOT / outputs_cfg["plot_dir"] / f"{rnn_type}_curves.png"
        plot_history(history, title=title, output_path=plot_path)
        print(f"Saved curves to: {plot_path}")

        for split_name in ["validation", "test"]:
            report_path = PROJECT_ROOT / outputs_cfg["report_dir"] / f"{rnn_type}_{split_name}_classification_report.txt"
            report = save_classification_report(
                model=model,
                dataloader=dataloaders[split_name],
                device=device,
                label_names=label_names,
                output_path=report_path,
                label_pad_id=train_cfg["label_pad_id"],
            )
            print(f"\n{title} {split_name} classification_report:")
            print(report)
            print(f"Saved report to: {report_path}")

        best_val_acc = max(history["val_accuracy"])
        summary_rows.append((rnn_type, best_val_acc, str(save_path)))

    print("\n=== Part 3(d): Result summary ===")
    for rnn_type, best_val_acc, save_path in summary_rows:
        print(f"{rnn_type.upper():>4} | best validation accuracy: {best_val_acc:.4f} | checkpoint: {save_path}")

if __name__ == "__main__":
    main()
