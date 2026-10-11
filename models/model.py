"""Bidirectional encoder models for token-level sequence labeling."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

RNNType = Literal["rnn", "lstm", "gru"]


class EncoderTagger(nn.Module):
    """Embedding + bidirectional recurrent encoder + linear classifier.

    The same class supports simple RNN, LSTM, and GRU by changing rnn_type.
    Output shape is [batch_size, sequence_length, num_labels].
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_size: int = 64,
        hidden_size: int = 64,
        rnn_type: RNNType = "lstm",
        num_layers: int = 1,
        bidirectional: bool = True,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in {"rnn", "lstm", "gru"}:
            raise ValueError("rnn_type must be one of: rnn, lstm, gru")

        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_token_id,
        )

        rnn_cls = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[rnn_type]

        self.encoder = rnn_cls(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * directions, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids)

        if lengths is not None:
            packed = pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, _ = self.encoder(packed)
            outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=input_ids.size(1),
            )
        else:
            outputs, _ = self.encoder(embedded)

        logits = self.classifier(outputs)
        return logits
