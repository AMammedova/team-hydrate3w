"""Member 3, W3.1/W3.4 — classification head (3-class) and reconstruction
head (masked-value regression), attached to the shared TCN/GRU encoder."""
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, encoded):
        return self.linear(encoded)


class ReconstructionHead(nn.Module):
    """Predicts the original (pre-mask) values at masked positions only."""
    def __init__(self, in_features: int, n_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, n_channels)

    def forward(self, encoded):
        raise NotImplementedError  # TODO: per-timestep reconstruction, not pooled
