"""
Module 5 — TCN. See project statement section 8 (DL5.1-DL5.4) and
Addendum A.2/A.3 for the reconciled channels-first + explicit-mask contract.

Reminder from the project statement: with the defaults below, the
receptive field is only ~1 timestep larger than the default window_size.
Always assert tcn.receptive_field() >= window_size in the training
script after any change to either -- see DL5.1's callout box.

Input convention (team contract): x and mask both arrive as
[batch, n_channels, window_size] (channels-first, no transpose needed --
this is Conv1d's native layout). mask is concatenated onto x along the
channel dimension before the first conv, so a model built with
n_channels=20 raw variables actually consumes 40 input channels.
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
        Each TemporalBlock stacks TWO causal convs at the same dilation,
        so each block contributes 2 * (kernel_size - 1) * dilation to the
        receptive field, not just one. With the class defaults
        (channel_sizes length 4 -> dilations [1,2,4,8], kernel_size=3)
        this works out to 61 -- see the project statement's callout
        about this margin being only ~1 timestep over window_size=60.
        """
        return 1 + sum(2 * (self.kernel_size - 1) * d for d in self.dilations)
