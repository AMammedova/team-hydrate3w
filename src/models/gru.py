"""
Module 6 — GRU. See project statement section 9 (DL6.1-DL6.3) and
Addendum A.2/A.3 for the reconciled channels-first + explicit-mask contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,             # RAW variable count; mask doubles this internally
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 3,
        bidirectional: bool = False,   # keep False -- causality, see DL6.1
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError(
                "bidirectional=True looks into the future within the window, "
                "which is invalid for a causal early-warning model (DL6.1). "
                "Pass a documented override only if you have a specific, "
                "reported reason to break causality."
            )
        self.gru = nn.GRU(
            input_size=n_channels * 2,   # raw values + mask, concatenated
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x, mask: (batch, n_channels, seq_len) -- channels-first, team contract.
        # nn.GRU wants (batch, seq_len, features), so transpose after concatenation.
        inp = torch.cat([x, mask.float()], dim=1).transpose(1, 2)  # (batch, seq_len, 2*n_channels)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                inp, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(inp)
        last_layer_hidden = h_n[-1]  # (batch, hidden_size)
        return self.head(last_layer_hidden)
