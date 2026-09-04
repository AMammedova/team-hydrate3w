"""Member 4 -- final TCN/GRU training runner (GPU target).

Runs the guaranteed Result-1 deep-model matrix:
    TCN x {real_only, real_plus_sim}
    GRU x {real_only, real_plus_sim}

Important design choices:
- real train/validation/test wells are generated ONCE with Member 2's
  GroupedKFoldSplitter; the real_plus_sim condition adds simulated rows only
  to the same real training fold, so condition comparisons are paired;
- validation/test never contain simulated instances;
- early stopping is validation PR-AUC through Trainer;
- validation AND test probabilities are saved for Member 5, because threshold
  selection belongs on validation and final event metrics belong on test;
- CUDA is the default and expected final environment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.contract import CONDITION_REAL_ONLY, CONDITION_REAL_PLUS_SIM, RESULTS_COLUMNS
from src.data.splits import GroupedKFoldSplitter, load_cache
from src.models.dataset import WindowDataset
from src.models.gru import GRUClassifier
from src.models.losses import weighted_cross_entropy
from src.models.tcn import TCN
from src.models.train_loop import Trainer, fold_seed, seed_everything, seed_worker


def _append_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _build_model(name: str, n_channels: int, window_size: int) -> torch.nn.Module:
    if name == "tcn":
        model = TCN(n_channels=n_channels, num_classes=3)
        if model.receptive_field() < window_size:
            raise ValueError(
                f"TCN receptive field {model.receptive_field()} < window size {window_size}"
            )
        return model
    if name == "gru":
        return GRUClassifier(n_channels=n_channels, num_classes=3)
    raise ValueError(f"unknown model {name!r}")


def _loader(
    ds: WindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def _save_outputs(
    path: Path,
    probs: np.ndarray,
    ds: WindowDataset,
) -> None:
    """Save everything Member 5 needs to reconstruct chronological alarms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    group_keys = np.asarray(list(ds.hours_by_well.keys()), dtype=np.int64)
    group_hours = np.asarray([ds.hours_by_well[g] for g in group_keys], dtype=np.float64)
    np.savez_compressed(
        path,
        probs=probs.astype(np.float32),
        y_true=ds.y.astype(np.int64),
        group=ds.group.astype(np.int64),
        inst_id=ds.inst_id.astype(np.int64),
        t_end=ds.t_end.astype(np.float64),
        is_sim=ds.is_sim.astype(np.uint8),
        failure_time=ds.failure_time.astype(np.float64),
        blockage_time=ds.blockage_time.astype(np.float64),
        normal_hours_group=group_keys,
        normal_hours_value=group_hours,
    )


def _base_real_splits(
    ds: WindowDataset,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    min_val_normal_hours: float,
    min_test_normal_hours: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create real-only folds once; simulated rows are added later only to train."""
    splitter = GroupedKFoldSplitter(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        include_sim_in_train=False,
        min_val_normal_hours=min_val_normal_hours,
        min_test_normal_hours=min_test_normal_hours,
    )
    splits = list(
        splitter.split(
            ds.X,
            ds.y,
            ds.group,
            is_sim=ds.is_sim,
            instances=ds.inst_id,
            well_hours=ds.hours_by_well,
        )
    )
    sim_rows = ds.is_sim.astype(bool)
    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
        if sim_rows[val_idx].any() or sim_rows[test_idx].any() or sim_rows[train_idx].any():
            raise AssertionError(f"fold {fold}: real-only split unexpectedly contains simulated rows")
        train_wells = set(ds.group[train_idx].tolist())
        val_wells = set(ds.group[val_idx].tolist())
        test_wells = set(ds.group[test_idx].tolist())
        if train_wells & val_wells or train_wells & test_wells or val_wells & test_wells:
            raise AssertionError(f"fold {fold}: well leakage detected")
    return splits


def run_one(
    *,
    model_name: str,
    condition: str,
    fold_idx: int,
    base_seed: int,
    ds: WindowDataset,
    train_idx_real: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    sim_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    grad_accum_steps: int,
    num_workers: int,
    lr: float,
    amp: bool,
    checkpoint_root: Path,
    outputs_dir: Path,
    resume: bool,
) -> list[dict]:
    run_seed = fold_seed(base_seed, fold_idx)
    seed_everything(run_seed)

    if condition == CONDITION_REAL_ONLY:
        train_idx = train_idx_real
    elif condition == CONDITION_REAL_PLUS_SIM:
        train_idx = np.concatenate([train_idx_real, sim_idx])
    else:
        raise ValueError(f"unknown condition {condition!r}")

    train_ds = ds.subset(train_idx)
    val_ds = ds.subset(val_idx)
    test_ds = ds.subset(test_idx)

    # Final guardrail: sim augmentation changes TRAIN only.
    if val_ds.is_sim.any() or test_ds.is_sim.any():
        raise AssertionError("simulated windows leaked into validation/test")

    train_loader = _loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=run_seed,
        device=device,
        num_workers=num_workers,
    )
    val_loader = _loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=run_seed,
        device=device,
        num_workers=num_workers,
    )
    test_loader = _loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=run_seed,
        device=device,
        num_workers=num_workers,
    )

    model = _build_model(model_name, ds.n_channels, ds.window_size)
    counts = torch.as_tensor(train_ds.class_counts(), dtype=torch.float32)
    loss_fn = weighted_cross_entropy(counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_dir = checkpoint_root / f"{model_name}_{condition}_fold{fold_idx}_seed{base_seed}"
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        amp=amp,
        grad_accum_steps=grad_accum_steps,
        checkpoint_dir=ckpt_dir,
    )
    history = trainer.fit(
        train_loader,
        val_loader,
        max_epochs=max_epochs,
        patience=patience,
        log_path=ckpt_dir / "train_log.jsonl",
        resume=resume,
    )
    trainer.load_best_checkpoint()

    val_probs = trainer.predict_proba(val_loader)
    test_probs = trainer.predict_proba(test_loader)
    stem = f"{model_name}_{condition}_fold{fold_idx}_seed{base_seed}"
    _save_outputs(outputs_dir / f"{stem}_val.npz", val_probs, val_ds)
    _save_outputs(outputs_dir / f"{stem}_test.npz", test_probs, test_ds)

    return [
        {
            "model": model_name,
            "fold": fold_idx,
            "seed": base_seed,
            "condition": condition,
            "metric_name": "best_val_pr_auc",
            "value": history["best_val_pr_auc"],
        },
        {
            "model": model_name,
            "fold": fold_idx,
            "seed": base_seed,
            "condition": condition,
            "metric_name": "best_epoch",
            "value": history["best_epoch"],
        },
        {
            "model": model_name,
            "fold": fold_idx,
            "seed": base_seed,
            "condition": condition,
            "metric_name": "n_epochs_trained",
            "value": len(history["epochs"]),
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out-results", default="results/results.csv")
    ap.add_argument("--outputs-dir", default="results/model_outputs")
    ap.add_argument("--checkpoint-root", default="checkpoints")
    ap.add_argument("--models", default="tcn,gru")
    ap.add_argument("--conditions", default=f"{CONDITION_REAL_ONLY},{CONDITION_REAL_PLUS_SIM}")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--min-val-normal-hours", type=float, default=300.0)
    ap.add_argument("--min-test-normal-hours", type=float, default=300.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but torch.cuda.is_available() is False. "
            "Run this script on the assigned GPU environment."
        )

    X, mask, index = load_cache(args.cache)
    ds = WindowDataset.from_arrays_and_index(X, mask, index)
    splits = _base_real_splits(
        ds,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.split_seed,
        min_val_normal_hours=args.min_val_normal_hours,
        min_test_normal_hours=args.min_test_normal_hours,
    )
    sim_idx = np.flatnonzero(ds.is_sim == 1)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print(
        f"device={device}; windows={len(ds)}; C={ds.n_channels}; W={ds.window_size}; "
        f"sim_windows={len(sim_idx)}; folds={len(splits)}; amp={not args.no_amp}"
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    all_rows: list[dict] = []
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        for condition in conditions:
            for model_name in models:
                for seed in seeds:
                    print(f"[fold {fold_idx}] {model_name} / {condition} / seed {seed}")
                    rows = run_one(
                        model_name=model_name,
                        condition=condition,
                        fold_idx=fold_idx,
                        base_seed=seed,
                        ds=ds,
                        train_idx_real=train_idx,
                        val_idx=val_idx,
                        test_idx=test_idx,
                        sim_idx=sim_idx,
                        device=device,
                        batch_size=args.batch_size,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        grad_accum_steps=args.grad_accum_steps,
                        num_workers=args.num_workers,
                        lr=args.lr,
                        amp=not args.no_amp,
                        checkpoint_root=Path(args.checkpoint_root),
                        outputs_dir=Path(args.outputs_dir),
                        resume=args.resume,
                    )
                    all_rows.extend(rows)
                    _append_results(Path(args.out_results), rows)

    print(f"wrote {len(all_rows)} training-diagnostic rows to {args.out_results}")
    print(f"saved validation/test probability files under {args.outputs_dir}")


if __name__ == "__main__":
    main()
