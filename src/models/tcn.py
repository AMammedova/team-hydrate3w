"""
Temporal Convolutional Network (TCN)

Implements a causal TCN classifier with explicit mask handling.

Note:
- With defaults, receptive field ≈ window_size + 1.
- Always assert tcn.receptive_field() >= window_size in training scripts.
- Input convention: x and mask arrive as (batch, n_channels, window_size).
  Mask is concatenated with x along the channel dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal: pad left only, in forward()
        self.pad = pad
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def _causal_conv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self._causal_conv(self.conv1, x)))
        out = self.dropout(self.relu(self._causal_conv(self.conv2, out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(
        self,
        n_channels: int,             # RAW variable count; mask doubles this internally
        num_classes: int = 3,
        channel_sizes: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channel_sizes = channel_sizes or [32, 32, 32, 32]
        self.kernel_size = kernel_size
        self.dilations = [2 ** i for i in range(len(channel_sizes))]

        layers = []
        in_ch = n_channels * 2   # raw values + mask, concatenated on the channel axis
        for out_ch, dilation in zip(channel_sizes, self.dilations):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x, mask: (batch, n_channels, seq_len) -- channels-first, team contract.
        inp = torch.cat([x, mask.float()], dim=1)   # (batch, 2*n_channels, seq_len)
        out = self.tcn(inp)                          # (batch, channels, seq_len)
        last = out[:, :, -1]                          # last-timestep representation
        return self.head(last)

    def receptive_field(self) -> int:
        """
        Each TemporalBlock stacks two causal convs at the same dilation,
        so each block contributes 2 * (kernel_size - 1) * dilation to the
        receptive field, not just one. With the class defaults
        (channel_sizes length 4 -> dilations [1,2,4,8], kernel_size=3)
        this works out to 61 -- this margin being only ~1 timestep over window_size=60.
        """
        return 1 + sum(2 * (self.kernel_size - 1) * d for d in self.dilations)
