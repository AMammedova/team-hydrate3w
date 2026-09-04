"""
Member 4 -- PyTorch adapter for cached hydrate windows.

The authoritative real-cache row order is owned by src.data.splits.load_cache().
Use WindowDataset.from_cache() in experiment code so GroupedKFoldSplitter indices
and PyTorch rows cannot silently drift apart.

WindowDataset.from_directory() is retained only as a lightweight compatibility
loader for legacy/fake caches used by sanity checks. Final experiments should use
from_cache().
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.splits import CacheIndex, load_cache


class WindowDataset(Dataset):
    """Flat, indexable dataset yielding ``(x, mask, y)``.

    Shapes follow the shared contract:
        X, mask: [N, C, W]
        y:       [N]

    Metadata are retained as public NumPy arrays so split/evaluation code can
    save validation/test outputs without reaching into private attributes.
    """

    def __init__(
        self,
        X: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        group: np.ndarray | None = None,
        inst_id: np.ndarray | None = None,
        t_end: np.ndarray | None = None,
        is_sim: np.ndarray | None = None,
        failure_time: np.ndarray | None = None,
        blockage_time: np.ndarray | None = None,
        hours_by_well: dict | None = None,
    ) -> None:
        if X is None:
            self.X = np.zeros((0, 0, 0), dtype=np.float32)
            self.mask = np.zeros((0, 0, 0), dtype=np.uint8)
            self.y = np.zeros(0, dtype=np.int64)
        else:
            if mask is None or y is None:
                raise ValueError("mask and y are required when X is provided")
            self.X = np.asarray(X, dtype=np.float32)
            self.mask = np.asarray(mask, dtype=np.uint8)
            self.y = np.asarray(y, dtype=np.int64)
            self._validate_core_shapes()

        n = len(self.y)
        self.group = self._meta_or_default(group, n, np.int64, 0)
        self.inst_id = self._meta_or_default(inst_id, n, np.int64, -1)
        self.t_end = self._meta_or_default(t_end, n, np.float64, np.nan)
        self.is_sim = self._meta_or_default(is_sim, n, np.uint8, 0)
        self.failure_time = self._meta_or_default(failure_time, n, np.float64, np.nan)
        self.blockage_time = self._meta_or_default(blockage_time, n, np.float64, np.nan)
        self.hours_by_well = dict(hours_by_well or {})

    @staticmethod
    def _meta_or_default(arr, n: int, dtype, fill):
        if arr is None:
            return np.full(n, fill, dtype=dtype)
        out = np.asarray(arr, dtype=dtype)
        if len(out) != n:
            raise ValueError(f"metadata length {len(out)} does not match dataset length {n}")
        return out

    def _validate_core_shapes(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must be [N,C,W], got {self.X.shape}")
        if self.mask.shape != self.X.shape:
            raise ValueError(f"mask shape {self.mask.shape} must equal X shape {self.X.shape}")
        if self.y.ndim != 1 or len(self.y) != len(self.X):
            raise ValueError(f"y must be [N] with N={len(self.X)}, got {self.y.shape}")

    @classmethod
    def from_cache(cls, cache_dir: str | Path) -> "WindowDataset":
        """Load the final cache in Member 2's authoritative row order."""
        X, mask, index = load_cache(cache_dir)
        return cls.from_arrays_and_index(X, mask, index)

    @classmethod
    def from_arrays_and_index(
        cls, X: np.ndarray, mask: np.ndarray, index: CacheIndex
    ) -> "WindowDataset":
        if len(X) != len(index) or len(mask) != len(index):
            raise ValueError("X/mask/CacheIndex lengths must match")
        return cls(
            X,
            mask,
            index.y,
            group=index.group,
            inst_id=index.inst_id,
            t_end=index.t_end,
            is_sim=index.is_sim,
            failure_time=index.failure_time,
            blockage_time=index.blockage_time,
            hours_by_well=index.hours_by_well,
        )

    @classmethod
    def from_directory(cls, cache_dir: str | Path) -> "WindowDataset":
        """Compatibility loader for fake/legacy cache files.

        Final experiments should use :meth:`from_cache`. This method tolerates
        older fake caches that do not yet contain ``blockage_time``.
        """
        paths = sorted(Path(cache_dir).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no .npz files found under {cache_dir}")

        Xs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        groups: list[np.ndarray] = []
        insts: list[np.ndarray] = []
        tends: list[np.ndarray] = []
        sims: list[np.ndarray] = []
        fails: list[np.ndarray] = []
        blocks: list[np.ndarray] = []
        hours_by_well: dict[int, float] = {}

        for path in paths:
            with np.load(path) as z:
                n = len(z["y"])
                if n == 0:
                    continue
                Xs.append(z["X"])
                masks.append(z["mask"])
                ys.append(z["y"])
                groups.append(z["group"])
                insts.append(z["inst_id"])
                tends.append(z["t_end"])
                sims.append(z["is_sim"])
                fails.append(np.full(n, float(z["failure_time"]), dtype=np.float64))
                block = float(z["blockage_time"]) if "blockage_time" in z.files else np.nan
                blocks.append(np.full(n, block, dtype=np.float64))
                g = int(z["group"][0])
                hours_by_well[g] = hours_by_well.get(g, 0.0) + float(z["normal_hours"])

        if not Xs:
            return cls()
        return cls(
            np.concatenate(Xs),
            np.concatenate(masks),
            np.concatenate(ys),
            group=np.concatenate(groups),
            inst_id=np.concatenate(insts),
            t_end=np.concatenate(tends),
            is_sim=np.concatenate(sims),
            failure_time=np.concatenate(fails),
            blockage_time=np.concatenate(blocks),
            hours_by_well=hours_by_well,
        )

    def subset(self, idx: Sequence[int] | np.ndarray) -> "WindowDataset":
        idx = np.asarray(idx, dtype=np.int64)
        out = WindowDataset(
            self.X[idx],
            self.mask[idx],
            self.y[idx],
            group=self.group[idx],
            inst_id=self.inst_id[idx],
            t_end=self.t_end[idx],
            is_sim=self.is_sim[idx],
            failure_time=self.failure_time[idx],
            blockage_time=self.blockage_time[idx],
            hours_by_well={
                g: h for g, h in self.hours_by_well.items() if g in set(self.group[idx].tolist())
            },
        )
        return out

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.mask[i]),
            torch.tensor(self.y[i], dtype=torch.long),
        )

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 3 else 0

    @property
    def window_size(self) -> int:
        return int(self.X.shape[2]) if self.X.ndim == 3 else 0

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y, minlength=3).astype(np.int64)

    def total_normal_hours(self) -> float:
        return float(sum(self.hours_by_well.values()))
