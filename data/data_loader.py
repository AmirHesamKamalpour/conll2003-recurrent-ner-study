"""Data loading, vocabulary, Dataset, and dynamic padding for CoNLL-2003 NER."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import DatasetDict, load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Please install Hugging Face datasets: pip install datasets"
    ) from exc


@dataclass(frozen=True)
class SpecialTokens:
    """Container for special vocabulary tokens."""

    pad: str = "<pad>"
    unk: str = "<unk>"
    start: str = "<s>"
    end: str = "</s>"

    def ordered(self) -> List[str]:
        return [self.pad, self.unk, self.start, self.end]


class Vocab:
    """Token vocabulary for an already-tokenized dataset.

    Required assignment functions:
    - build_vocab
    - token_to_id
    - id_to_token
    - vocab_size
    """

    def __init__(self, special_tokens: SpecialTokens | None = None) -> None:
        self.special_tokens = special_tokens or SpecialTokens()
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        self._add_special_tokens()

    def _add_token(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def _add_special_tokens(self) -> None:
        for token in self.special_tokens.ordered():
            self._add_token(token)

    def build_vocab(self, token_sequences: Iterable[Sequence[str]], cutoff: int = 1) -> None:
        """Build vocabulary from tokenized sentences.

        Tokens with frequency lower than cutoff are skipped. The dataset is already
        tokenized, so this method expects a sequence of tokens per sentence.
        """
        if cutoff < 1:
            raise ValueError("cutoff must be >= 1")

        counts: Counter[str] = Counter()
        for sentence in token_sequences:
            counts.update(sentence)

        # Keep a deterministic order: frequency descending, then alphabetical.
        # Special tokens keep ids 0..3 because they were inserted before building.
        for token, frequency in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if frequency >= cutoff:
                self._add_token(token)

    def token_to_id(self, token: str) -> int:
        """Convert token to integer id, returning <unk> for unseen tokens."""
        return self.stoi.get(token, self.stoi[self.special_tokens.unk])

    def id_to_token(self, idx: int) -> str:
        """Convert integer id to token string."""
        if idx < 0 or idx >= len(self.itos):
            raise IndexError(f"Token id {idx} is outside the vocabulary range.")
        return self.itos[idx]

    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.token_to_id(self.special_tokens.pad)

    @property
    def unk_id(self) -> int:
        return self.token_to_id(self.special_tokens.unk)


class NERSequenceDataset(Dataset):
    """Torch Dataset for token-level NER sequence labeling."""

    def __init__(
        self,
        hf_split,
        vocab: Vocab,
        token_column: str = "tokens",
        label_column: str = "ner_tags",
    ) -> None:
        self.hf_split = hf_split
        self.vocab = vocab
        self.token_column = token_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        example = self.hf_split[idx]
        token_ids = torch.tensor(
            [self.vocab.token_to_id(token) for token in example[self.token_column]],
            dtype=torch.long,
        )
        label_ids = torch.tensor(example[self.label_column], dtype=torch.long)
        length = len(token_ids)
        return token_ids, label_ids, length


class SequenceCollator:
    """Dynamic batch padding with separate pad values for tokens and labels."""

    def __init__(self, token_pad_id: int, label_pad_id: int = -100) -> None:
        self.token_pad_id = token_pad_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch):
        token_ids, label_ids, lengths = zip(*batch)
        padded_tokens = pad_sequence(
            token_ids,
            batch_first=True,
            padding_value=self.token_pad_id,
        )
        padded_labels = pad_sequence(
            label_ids,
            batch_first=True,
            padding_value=self.label_pad_id,
        )
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        return {
            "input_ids": padded_tokens,
            "labels": padded_labels,
            "lengths": lengths_tensor,
        }


def load_conll_dataset(
    dataset_name: str,
    revision: str | None = "convert/parquet",
) -> DatasetDict:
    """Load the CoNLL dataset from Hugging Face.

    Newer versions of the `datasets` library no longer execute Python dataset
    loading scripts. The eriktks/conll2003 repository still has conll2003.py on
    the main branch, so we load the Hugging Face Parquet conversion branch.
    """
    if revision:
        try:
            return load_dataset(dataset_name, revision=revision)
        except Exception as exc:
            print(
                f"Could not load {dataset_name!r} with revision={revision!r}: {exc}\n"
                "Trying the default repository revision next..."
            )

    # This path works with older datasets versions that still allow the original
    # loading script. It is intentionally tried second to avoid the datasets>=4
    # script error when a Parquet conversion is available.
    try:
        return load_dataset(dataset_name)
    except RuntimeError as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" in message or "trust_remote_code" in message:
            fallback_name = "lhoestq/conll2003"
            print(
                f"Default load failed because the dataset uses a loading script. "
                f"Trying script-free fallback dataset: {fallback_name!r}."
            )
            return load_dataset(fallback_name)
        raise


def build_vocab_from_train_split(
    train_split,
    token_column: str = "tokens",
    cutoff: int = 1,
    special_tokens: SpecialTokens | None = None,
) -> Vocab:
    """Build Vocab from only the training split, as required."""
    vocab = Vocab(special_tokens=special_tokens)
    vocab.build_vocab(train_split[token_column], cutoff=cutoff)
    return vocab


DEFAULT_NER_LABEL_NAMES = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def get_label_names(dataset_dict: DatasetDict, label_column: str = "ner_tags") -> List[str]:
    """Return label names for a sequence label column.

    The original script-based dataset exposes ClassLabel names in Features.
    The Parquet conversion may expose only integer sequences, so we fall back to
    the standard CoNLL-2003 NER label order.
    """
    label_feature = dataset_dict["train"].features[label_column]
    if hasattr(label_feature, "feature") and hasattr(label_feature.feature, "names"):
        return list(label_feature.feature.names)
    if label_column == "ner_tags":
        return DEFAULT_NER_LABEL_NAMES
    raise ValueError(f"Could not extract label names from column: {label_column}")


def make_dataloaders(
    dataset_dict: DatasetDict,
    vocab: Vocab,
    batch_size: int,
    token_column: str = "tokens",
    label_column: str = "ner_tags",
    label_pad_id: int = -100,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create train/validation/test DataLoaders with collate_fn padding."""
    collator = SequenceCollator(token_pad_id=vocab.pad_id, label_pad_id=label_pad_id)

    dataloaders: Dict[str, DataLoader] = {}
    for split in ["train", "validation", "test"]:
        if split not in dataset_dict:
            continue
        ds = NERSequenceDataset(
            dataset_dict[split],
            vocab=vocab,
            token_column=token_column,
            label_column=label_column,
        )
        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collator,
            num_workers=num_workers,
        )
    return dataloaders


def describe_dataset(dataset_dict: DatasetDict) -> str:
    """Return a compact dataset-analysis string for Part 1(a)."""
    lines = []
    lines.append("Dataset splits and sizes:")
    for split_name, split_data in dataset_dict.items():
        lines.append(f"  - {split_name}: {len(split_data)} examples")
    lines.append("Columns/features:")
    for column_name, feature in dataset_dict["train"].features.items():
        lines.append(f"  - {column_name}: {feature}")
    return "\n".join(lines)
