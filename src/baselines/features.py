"""Multi-timescale rolling features over causal windows."""

from __future__ import annotations

import numpy as np

STATS_DEFAULT = ("mean", "std", "min", "max", "slope", "last_diff")
SCALES_DEFAULT = (1.0, 0.5, 0.25)
PRESENCE_SUFFIX = "presence_frac"

_NEEDS_TWO = frozenset({"std", "slope", "last_diff"})


class RollingFeatureExtractor:
    """3 timescales (taken from the END of the window, so features stay causal)
    x 6 statistics per channel, plus one presence fraction per channel. All
    statistics use present (mask==1) samples only.

    missing_policy -- a channel that observed nothing is "nan" (XGBoost learns
        a default branch direction) or "zero" (legacy). 0.0 is not neutral:
        after per-instance normalisation it is the value a healthy sensor at
        its own baseline reports.
    slope_time -- "index" regresses on true position so gaps keep their width
        and the slope stays a rate; "rank" (legacy) closes gaps.
    """

    def __init__(
        self,
        stats: list[str] | None = None,
        scales: list[float] | None = None,
        missing_policy: str = "nan",
        slope_time: str = "index",
    ) -> None:
        self.stats = list(stats) if stats else list(STATS_DEFAULT)
        self.scales = list(scales) if scales else list(SCALES_DEFAULT)

        unknown = [s for s in self.stats if s not in STATS_DEFAULT]
        if unknown:
            raise ValueError(f"Unknown stat(s): {unknown!r}; known: {list(STATS_DEFAULT)}")
        if missing_policy not in ("nan", "zero"):
            raise ValueError(f"missing_policy must be 'nan' or 'zero', got {missing_policy!r}")
        if slope_time not in ("index", "rank"):
            raise ValueError(f"slope_time must be 'index' or 'rank', got {slope_time!r}")
        if any(not (0.0 < s <= 1.0) for s in self.scales):
            raise ValueError(f"scales must all be in (0, 1], got {self.scales!r}")
        self.missing_policy = missing_policy
        self.slope_time = slope_time

    def n_features(self, n_channels: int) -> int:
        return n_channels * len(self.stats) * len(self.scales) + n_channels

    def feature_names(
        self,
        channels: list[str] | None = None,
        n_channels: int | None = None,
    ) -> list[str]:
        """Column names in transform()'s exact order.

        Grammar "<channel>|<stat>|scale<f>" / "<channel>|presence_frac". The
        separator is "|" because 3W channel names contain "-" and "last_diff"
        contains "_". Raises rather than inventing names.
        """
        if channels is None:
            if n_channels is None:
                raise ValueError("pass channels=[...] or n_channels=<int>")
            channels = [f"ch{i}" for i in range(n_channels)]
        elif n_channels is not None and len(channels) != n_channels:
            raise ValueError(f"channels has {len(channels)} entries but n_channels={n_channels}")

        names: list[str] = []
        for scale in self.scales:                 # same nesting as transform()
            for ch in channels:
                for stat in self.stats:
                    names.append(f"{ch}|{stat}|scale{scale}")
        for ch in channels:
            names.append(f"{ch}|{PRESENCE_SUFFIX}")
        return names

    def _slice_stats(self, Xs: np.ndarray, Ms: np.ndarray) -> dict:
        m = Ms.astype(bool)
        Xf = np.asarray(Xs, dtype=np.float64)
        xz = np.where(m, Xf, 0.0)

        cnt = m.sum(axis=-1)
        safe = np.maximum(cnt, 1)

        out: dict[str, np.ndarray] = {}
        mean = xz.sum(axis=-1) / safe
        need = set(self.stats)

        if "mean" in need:
            out["mean"] = mean
        if "std" in need:
            var = (xz * xz).sum(axis=-1) / safe - mean * mean
            out["std"] = np.sqrt(np.maximum(var, 0.0))
        if "min" in need:
            out["min"] = np.where(m, Xf, np.inf).min(axis=-1)
        if "max" in need:
            out["max"] = np.where(m, Xf, -np.inf).max(axis=-1)

        if "slope" in need:
            S = Xf.shape[-1]
            if self.slope_time == "rank":
                t_raw = np.cumsum(m, axis=-1) - 1.0
            else:
                t_raw = np.broadcast_to(np.arange(S, dtype=np.float64), Xf.shape)
            t_present = np.where(m, t_raw, 0.0)
            t_bar = t_present.sum(axis=-1) / safe
            tc = np.where(m, t_raw - t_bar[..., None], 0.0)
            denom = (tc * tc).sum(axis=-1)
            num = (tc * xz).sum(axis=-1)
            out["slope"] = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)

        if "last_diff" in need:
            first_i = np.argmax(m, axis=-1)
            last_i = m.shape[-1] - 1 - np.argmax(m[..., ::-1], axis=-1)
            first_v = np.take_along_axis(Xf, first_i[..., None], axis=-1)[..., 0]
            last_v = np.take_along_axis(Xf, last_i[..., None], axis=-1)[..., 0]
            out["last_diff"] = last_v - first_v

        empty = cnt == 0
        singleton = cnt < 2
        fill = np.nan if self.missing_policy == "nan" else 0.0
        for stat in list(out):
            vals = np.asarray(out[stat], dtype=np.float64)
            if stat in _NEEDS_TWO:
                # One sample really does imply zero spread and zero trend.
                vals = np.where(singleton & ~empty, 0.0, vals)
            out[stat] = np.where(empty, fill, vals)
        return out

    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """X, mask: (n_windows, n_channels, window_size), channels-first.

        Returns float32 (n_windows, n_features). The trailing n_channels block
        is the presence fraction over the full window and is never NaN.
        """
        X = np.asarray(X)
        mask = np.asarray(mask)
        if X.ndim != 3:
            raise ValueError(f"expected X with shape (N, C, W), got {X.shape}")
        if X.shape != mask.shape:
            raise ValueError(f"X {X.shape} and mask {mask.shape} must have the same shape")

        n_windows, n_channels, window_size = X.shape
        out = np.empty((n_windows, self.n_features(n_channels)), dtype=np.float32)

        col = 0
        for scale in self.scales:
            n_steps = max(1, int(round(scale * window_size)))
            stats = self._slice_stats(X[:, :, -n_steps:], mask[:, :, -n_steps:])
            for c in range(n_channels):
                for stat in self.stats:
                    out[:, col] = stats[stat][:, c]
                    col += 1

        out[:, col : col + n_channels] = mask.astype(np.float64).mean(axis=-1)
        col += n_channels

        assert col == out.shape[1], f"emitted {col} columns, expected {out.shape[1]}"
        return out
