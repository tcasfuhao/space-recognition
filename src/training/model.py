from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BoundaryModelConfig:
    vocab_size: int
    embedding_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2


class BoundaryBiLSTM(nn.Module):
    def __init__(self, config: BoundaryModelConfig):
        super().__init__()
        self.model_config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=input_ids.shape[1]
        )
        return self.classifier(self.dropout(encoded)).squeeze(-1)

    def serializable_config(self) -> dict[str, int | float]:
        return asdict(self.model_config)

