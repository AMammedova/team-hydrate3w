"""Class-imbalance-aware losses for the 3-class Normal/Transient/Established target."""

from __future__ import annotations

import torch
import torch.nn as nn


def weighted_cross_entropy(class_counts: torch.Tensor) -> nn.Module:
    """class_counts: (3,) counts of Normal/Transient/Established in the training fold."""
    weights = class_counts.sum() / (class_counts.clamp(min=1) * len(class_counts))
    return nn.CrossEntropyLoss(weight=weights)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()
