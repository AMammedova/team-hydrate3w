"""Unit tests for GPU-ready Trainer.

Tests run on CPU intentionally because they validate control flow, checkpointing,
metrics and dataset semantics; CUDA/AMP is exercised by the real GPU smoke run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dataset import WindowDataset
from src.models.gru import GRUClassifier
from src.models.train_loop import Trainer, fold_seed, seed_everything


def _easy_dataset(n: int = 90, c: int = 4, w: int = 12, seed: int = 0) -> WindowDataset:
    """Three learnable classes with guaranteed class coverage."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(3, dtype=np.int64), n // 3)
    if len(y) < n:
        y = np.concatenate([y, np.arange(n - len(y), dtype=np.int64)])
    X = rng.normal(0, 0.15, size=(n, c, w)).astype(np.float32)
    X += y[:, None, None].astype(np.float32) * 1.5
    mask = np.ones_like(X, dtype=np.uint8)
    group = np.repeat(np.arange(max(1, n // 10)), 10)[:n].astype(np.int64)
    inst = np.arange(n, dtype=np.int64)
    t_end = np.arange(n, dtype=np.float64)
    return WindowDataset(X, mask, y, group=group, inst_id=inst, t_end=t_end)


def _loaders(ds: WindowDataset):
    # Explicit stratified-ish slices so validation always has Normal + positives.
    train_idx = np.arange(0, 60)
    val_idx = np.arange(60, 90)
    return (
        DataLoader(ds.subset(train_idx), batch_size=12, shuffle=True),
        DataLoader(ds.subset(val_idx), batch_size=12, shuffle=False),
    )


def test_fit_runs_and_records_pr_auc(tmp_path):
    ds = _easy_dataset()
    train_loader, val_loader = _loaders(ds)
    seed_everything(42)
    model = GRUClassifier(n_channels=ds.n_channels, hidden_size=12, num_classes=3)
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-2),
        torch.nn.CrossEntropyLoss(),
        device="cpu",
        amp=False,
        checkpoint_dir=tmp_path / "ckpt",
    )
    history = trainer.fit(train_loader, val_loader, max_epochs=4, patience=10)
    assert len(history["epochs"]) == 4
    assert all(np.isfinite(e["val_pr_auc"]) for e in history["epochs"])
    assert history["best_epoch"] is not None


def test_early_stopping_respects_patience(tmp_path):
    ds = _easy_dataset()
    train_loader, val_loader = _loaders(ds)
    model = GRUClassifier(n_channels=ds.n_channels, hidden_size=8, num_classes=3)
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters(), lr=0.0),
        torch.nn.CrossEntropyLoss(),
        device="cpu",
        amp=False,
        checkpoint_dir=tmp_path / "ckpt",
    )
    patience = 3
    history = trainer.fit(train_loader, val_loader, max_epochs=100, patience=patience)
    assert len(history["epochs"]) <= patience + 1


def test_checkpoint_reload_and_resume_state(tmp_path):
    ds = _easy_dataset()
    train_loader, val_loader = _loaders(ds)
    ckpt_dir = tmp_path / "ckpt"
    model = GRUClassifier(n_channels=ds.n_channels, hidden_size=8, num_classes=3)
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        torch.nn.CrossEntropyLoss(),
        device="cpu",
        amp=False,
        checkpoint_dir=ckpt_dir,
    )
    trainer.fit(train_loader, val_loader, max_epochs=3, patience=10)
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "last.pt").exists()

    model2 = GRUClassifier(n_channels=ds.n_channels, hidden_size=8, num_classes=3)
    trainer2 = Trainer(
        model2,
        torch.optim.Adam(model2.parameters(), lr=1e-3),
        torch.nn.CrossEntropyLoss(),
        device="cpu",
        amp=False,
        checkpoint_dir=ckpt_dir,
    )
    trainer2.load_best_checkpoint()
    last_epoch = trainer2.load_last_checkpoint()
    assert last_epoch == 2


def test_predict_proba_shape_and_rows_sum_to_one(tmp_path):
    ds = _easy_dataset(n=30)
    loader = DataLoader(ds, batch_size=7, shuffle=False)
    model = GRUClassifier(n_channels=ds.n_channels, hidden_size=8, num_classes=3)
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters()),
        torch.nn.CrossEntropyLoss(),
        device="cpu",
        amp=False,
        checkpoint_dir=tmp_path / "ckpt",
    )
    probs = trainer.predict_proba(loader)
    assert probs.shape == (len(ds), 3)
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_gradient_accumulation_steps_final_partial_group(tmp_path):
    ds = _easy_dataset(n=30)
    loader = DataLoader(ds, batch_size=4, shuffle=False)  # 8 batches, accum=3 -> tail of 2
    model = GRUClassifier(n_channels=ds.n_channels, hidden_size=8, num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        opt,
        torch.nn.CrossEntropyLoss(),
        device="cpu",
        amp=False,
        grad_accum_steps=3,
        checkpoint_dir=tmp_path / "ckpt",
    )
    before = [p.detach().clone() for p in model.parameters()]
    trainer._run_epoch_train(loader)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_undefined_validation_pr_auc_fails_clearly(tmp_path):
    ds = _easy_dataset(n=30)
    train = DataLoader(ds.subset(np.arange(0, 20)), batch_size=5)
    normal_idx = np.flatnonzero(ds.y == 0)[:5]
    val = DataLoader(ds.subset(normal_idx), batch_size=5)
    model = GRUClassifier(n_channels=ds.n_channels, hidden_size=8, num_classes=3)
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        torch.nn.CrossEntropyLoss(),
        device="cpu",
        amp=False,
        checkpoint_dir=tmp_path / "ckpt",
    )
    try:
        trainer.fit(train, val, max_epochs=2, patience=1)
    except RuntimeError as exc:
        assert "validation PR-AUC was undefined" in str(exc)
    else:
        raise AssertionError("expected undefined validation PR-AUC to raise RuntimeError")


def test_fold_seed_is_deterministic_per_fold():
    assert fold_seed(42, 0) == 42
    assert fold_seed(42, 3) == 45
    assert fold_seed(42, 0) == fold_seed(42, 0)


def test_subset_preserves_row_alignment_and_is_disjoint():
    ds = _easy_dataset(n=30)
    a = ds.subset(np.arange(0, 15))
    b = ds.subset(np.arange(15, 30))
    a_keys = set(zip(a.inst_id.tolist(), a.t_end.tolist()))
    b_keys = set(zip(b.inst_id.tolist(), b.t_end.tolist()))
    assert a_keys.isdisjoint(b_keys)
    assert np.array_equal(a.y, ds.y[:15])
