"""
GPU-ready training loop for TCN/GRU.

Key guarantees:
- models receive ``model(x, mask)``;
- early stopping/checkpoint selection uses validation PR-AUC;
- class-weighted loss modules are moved to the same CUDA device as logits;
- AMP is enabled only on CUDA;
- gradient accumulation handles an incomplete final accumulation group;
- best/last checkpoints contain enough state for a safe resume;
- epoch metrics are written to disk as JSONL when requested.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.contract import NORMAL
from src.eval.metrics import positive_score, pr_auc


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Standard deterministic DataLoader worker seeding."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fold_seed(base_seed: int, fold_index: int) -> int:
    return int(base_seed + fold_index)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str | torch.device,
        amp: bool = True,
        grad_accum_steps: int = 1,
        checkpoint_dir: str | Path = "checkpoints/",
    ) -> None:
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, torch.nn.Module) else loss_fn
        self.amp = bool(amp and self.device.type == "cuda")
        self.grad_accum_steps = int(grad_accum_steps)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_score = float("-inf")
        self._best_epoch: int | None = None
        self._resume_epoch = -1

        # New torch API; safe because project requirements use a modern PyTorch.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else None,
            enabled=self.amp,
        )

    def _run_epoch_train(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        total_batches = len(loader)

        for step, (x, mask, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # The final accumulation group can contain fewer than
            # grad_accum_steps batches; divide by its actual size.
            group_start = (step // self.grad_accum_steps) * self.grad_accum_steps
            accum_divisor = min(self.grad_accum_steps, total_batches - group_start)

            with self._autocast():
                logits = self.model(x, mask)
                raw_loss = self.loss_fn(logits, y)
                loss = raw_loss / accum_divisor

            self.scaler.scale(loss).backward()

            end_of_group = ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == total_batches)
            if end_of_group:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(raw_loss.detach().item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _run_epoch_val(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_probs: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        for x, mask, y in loader:
            x = x.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self._autocast():
                logits = self.model(x, mask)
                loss = self.loss_fn(logits, y)

            total_loss += float(loss.item())
            n_batches += 1
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_y.append(y.cpu().numpy())

        val_loss = total_loss / max(n_batches, 1)
        if not all_probs:
            return val_loss, float("nan")

        probs = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_y, axis=0)
        y_bin = (y_true != NORMAL).astype(np.int64)
        if y_bin.min() == y_bin.max():
            return val_loss, float("nan")
        return val_loss, pr_auc(y_bin, positive_score(probs))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 10,
        log_path: str | Path | None = None,
        resume: bool = False,
    ) -> dict:
        if max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")

        if resume:
            self.load_last_checkpoint()
        start_epoch = self._resume_epoch + 1

        history: dict = {
            "epochs": [],
            "best_epoch": self._best_epoch,
            "best_val_pr_auc": self._best_val_score,
        }
        epochs_without_improvement = 0

        log_file = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf8")

        try:
            for epoch in range(start_epoch, max_epochs):
                train_loss = self._run_epoch_train(train_loader)
                val_loss, val_score = self._run_epoch_val(val_loader)

                score_defined = math.isfinite(val_score)
                is_best = score_defined and val_score > self._best_val_score

                if is_best:
                    self._best_val_score = float(val_score)
                    self._best_epoch = epoch
                    epochs_without_improvement = 0
                    self._save_checkpoint(epoch, val_score, tag="best")
                else:
                    epochs_without_improvement += 1

                self._save_checkpoint(epoch, val_score, tag="last")

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_pr_auc": val_score,
                    "is_best": bool(is_best),
                }
                history["epochs"].append(record)
                history["best_epoch"] = self._best_epoch
                history["best_val_pr_auc"] = self._best_val_score

                if log_file is not None:
                    log_file.write(json.dumps(record, allow_nan=True) + "\n")
                    log_file.flush()

                if epochs_without_improvement >= patience:
                    break
        finally:
            if log_file is not None:
                log_file.close()

        if self._best_epoch is None:
            raise RuntimeError(
                "validation PR-AUC was undefined for every epoch, so no best checkpoint "
                "could be selected. Check the validation fold: it must contain both "
                "Normal and positive (Transient/Established) windows."
            )
        return history

    def _save_checkpoint(self, epoch: int, val_score: float, tag: str) -> Path:
        path = self.checkpoint_dir / f"{tag}.pt"
        torch.save(
            {
                "epoch": int(epoch),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "val_pr_auc": float(val_score),
                "best_val_pr_auc": float(self._best_val_score),
                "best_epoch": self._best_epoch,
            },
            path,
        )
        return path

    def load_best_checkpoint(self) -> None:
        path = self.checkpoint_dir / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"no best checkpoint at {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self._best_val_score = float(ckpt.get("best_val_pr_auc", ckpt["val_pr_auc"]))
        self._best_epoch = ckpt.get("best_epoch", ckpt.get("epoch"))

    def load_last_checkpoint(self) -> int:
        path = self.checkpoint_dir / "last.pt"
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint to resume from at {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self._best_val_score = float(ckpt.get("best_val_pr_auc", float("-inf")))
        self._best_epoch = ckpt.get("best_epoch")
        self._resume_epoch = int(ckpt["epoch"])
        return self._resume_epoch

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_probs: list[np.ndarray] = []
        for x, mask, _y in loader:
            x = x.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(x, mask)
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        if not all_probs:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(all_probs, axis=0).astype(np.float32)
