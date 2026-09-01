"""
Module 7 — Training loop. See project statement section 10 (DL7.0-DL7.4)
and Addendum A.3 (models take (x, mask) explicitly, not pre-merged).

DL7.0: val_loader must be built from val_idx yielded by
GroupedKFoldSplitter.split() (src/data/splitting.py) -- never a random
row-level split -- or the well-level leakage control is silently
defeated.

Each batch from train_loader/val_loader is expected to be
(x, mask, y) -- x, mask channels-first (batch, n_channels, window_size),
per the team data contract -- so model calls throughout this class are
model(x, mask), not model(x).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str,
        amp: bool = True,
        grad_accum_steps: int = 1,
        checkpoint_dir: str = "checkpoints/",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.amp = amp
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._best_val_score = float("-inf")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 10,
    ) -> dict:
        # TODO: standard train/eval loop. Each batch is (x, mask, y);
        # forward pass is self.model(x, mask). Early-stop and checkpoint
        # on validation PR-AUC (DL7.1), not accuracy. Checkpoint every
        # epoch to self.checkpoint_dir (on NVMe scratch per the compute
        # plan) so a mid-training interruption can resume (DL7.2).
        raise NotImplementedError

    def predict_proba(self, loader: DataLoader) -> "np.ndarray":
        # TODO: iterate (x, mask, y) batches, call self.model(x, mask),
        # softmax, concatenate across the loader.
        raise NotImplementedError

    def load_best_checkpoint(self) -> None:
        # TODO
        raise NotImplementedError
