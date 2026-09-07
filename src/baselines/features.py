"""
Module 4 (features) — see project statement section 7, and Addendum A.7
for the multi-timescale + mask-aware requirements. Owner: Member 3.

Multi-timescale rolling features over causal windows:
  - 3 time scales: full window, last half, last quarter (taken from the END
    of the window, so every feature is causal)
  - 6 statistics per scale per channel: mean, std, min, max, slope, last_diff
  - 1 presence fraction per channel (full window)
  - All statistics computed over PRESENT (mask==1) samples only

Feature block order -- this is the contract feature_names() encodes, and the
only order transform() ever emits:

    for scale in scales:            # 1.0, 0.5, 0.25
        for channel in channels:
            for stat in stats:      # mean, std, min, max, slope, last_diff
    then one presence-fraction column per channel

feature_names() is generated from the same nested loop as transform() so the
two cannot drift. Getting this wrong is a silent failure: XGBoost would still
train, importances would still plot, and every physical attribution in the
report would name the wrong sensor.


TWO MEASURED DESIGN DECISIONS
-----------------------------
Ablated in tools/ablate_features.py; numbers land in
results/ablation_features.csv and are quoted in the report's Method section.

1. missing_policy -- what a feature is when a channel observed NOTHING in
   the slice.

   "nan" (default): emit NaN. XGBoost learns a default branch direction for
       NaN at every split, so "this sensor was dead" becomes usable signal.
   "zero" (legacy): emit 0.0.

   0.0 is actively misleading here. windowing.normalize_instance()
   standardises each instance to roughly zero mean, so a feature of 0.0 is
   the value a perfectly healthy sensor sitting at its own baseline reports.
   The legacy encoding therefore tells the model "this dead sensor looks
   completely normal" -- the worst available reading, because
   DATA_FINDINGS.md section 9 shows the frozen sensors are frozen precisely
   in the instances that carry events, including one of the three blockages.

2. slope_time -- the time axis the least-squares slope is regressed on.

   "index" (default): the sample's true position inside the slice, so a gap
       widens the x-spacing and the slope stays a rate per sample.
   "rank" (legacy): the sample's rank among present samples, which silently
       closes every gap and inflates the apparent rate of change.

   Rate of change is the physical hydrate signature -- a restriction builds,
   upstream pressure ramps -- so a biased slope attacks the one feature the
   domain argument in the Discussion leans on.

Both legacy behaviours stay reachable so the ablation is a real comparison
rather than a claim.
"""

from __future__ import annotations

import numpy as np

STATS_DEFAULT = ("mean", "std", "min", "max", "slope", "last_diff")
SCALES_DEFAULT = (1.0, 0.5, 0.25)
PRESENCE_SUFFIX = "presence_frac"

# Stats that need at least two present samples before they are defined.
_NEEDS_TWO = frozenset({"std", "slope", "last_diff"})


class RollingFeatureExtractor:
    def __init__(
        self,
        stats: list[str] | None = None,
        scales: list[float] | None = None,   # fraction of the window, from the end
        missing_policy: str = "nan",         # "nan" | "zero"
        slope_time: str = "index",           # "index" | "rank"
    ) -> None:
        self.stats = list(stats) if stats else list(STATS_DEFAULT)
        # full window, last half, last quarter -- per Addendum A.7 / team W2.1
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

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------
    def n_features(self, n_channels: int) -> int:
        return n_channels * len(self.stats) * len(self.scales) + n_channels

    def feature_names(
        self,
        channels: list[str] | None = None,
        n_channels: int | None = None,
    ) -> list[str]:
        """Column names for transform()'s output, in transform()'s exact order.

        Pass the real channel names (cache_config.json -> "channels") so
        importance.py can attribute a split back to a physical sensor. With
        neither argument this raises rather than inventing names: a
        wrong-but-plausible name list is worse than no name list.

        Name grammar, parsed by importance.summarize_importances:
            "<channel>|<stat>|scale<scale>"     e.g. "P-MON-CKP|slope|scale0.5"
            "<channel>|presence_frac"
        The separator is "|" because 3W channel names contain "-" and the stat
        name "last_diff" contains "_", so neither is safe.
        """
        if channels is None:
            if n_channels is None:
                raise ValueError("pass channels=[...] or n_channels=<int>")
            channels = [f"ch{i}" for i in range(n_channels)]
        elif n_channels is not None and len(channels) != n_channels:
            raise ValueError(f"channels has {len(channels)} entries but n_channels={n_channels}")

        names: list[str] = []
        for scale in self.scales:                 # SAME nesting as transform()
            for ch in channels:
                for stat in self.stats:
                    names.append(f"{ch}|{stat}|scale{scale}")
        for ch in channels:
            names.append(f"{ch}|{PRESENCE_SUFFIX}")
        return names

    # ------------------------------------------------------------------
    # Vectorised statistics
    # ------------------------------------------------------------------
    def _slice_stats(self, Xs: np.ndarray, Ms: np.ndarray) -> dict:
        """All requested statistics for one scale, for every window and
        channel at once.

        Xs, Ms: (N, C, S). Returns {stat: (N, C) float64}.

        Everything is masked arithmetic, with no Python loop over windows. On
        the real cache that is the difference between a feature matrix built
        in seconds and one built in tens of minutes, which is what makes an
        honest hyperparameter search affordable at all.
        """
        m = Ms.astype(bool)
        Xf = np.asarray(Xs, dtype=np.float64)
        xz = np.where(m, Xf, 0.0)               # zeroed so sums ignore absent

        cnt = m.sum(axis=-1)                    # (N, C)
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
                # Rank among present samples: 0, 1, 2, ... (legacy behaviour).
                t_raw = np.cumsum(m, axis=-1) - 1.0
            else:
                # True position inside the slice, so gaps keep their width.
                t_raw = np.broadcast_to(np.arange(S, dtype=np.float64), Xf.shape)
            t_present = np.where(m, t_raw, 0.0)
            t_bar = t_present.sum(axis=-1) / safe
            tc = np.where(m, t_raw - t_bar[..., None], 0.0)
            denom = (tc * tc).sum(axis=-1)
            num = (tc * xz).sum(axis=-1)
            out["slope"] = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)

        if "last_diff" in need:
            # Net change between the first and last PRESENT sample of the slice.
            first_i = np.argmax(m, axis=-1)
            last_i = m.shape[-1] - 1 - np.argmax(m[..., ::-1], axis=-1)
            first_v = np.take_along_axis(Xf, first_i[..., None], axis=-1)[..., 0]
            last_v = np.take_along_axis(Xf, last_i[..., None], axis=-1)[..., 0]
            out["last_diff"] = last_v - first_v

        # Undefined cases. An empty slice has no statistic at all; std, slope
        # and last_diff additionally need two points before they mean anything.
        empty = cnt == 0
        singleton = cnt < 2
        fill = np.nan if self.missing_policy == "nan" else 0.0
        for stat in list(out):
            vals = np.asarray(out[stat], dtype=np.float64)
            if stat in _NEEDS_TWO:
                # A one-sample slice has no spread and no trend: 0.0 is the
                # right answer there, not a missing one.
                vals = np.where(singleton & ~empty, 0.0, vals)
            out[stat] = np.where(empty, fill, vals)
        return out

    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        X, mask: (n_windows, n_channels, window_size) -- channels-first.

        Returns (n_windows, n_channels * len(stats) * len(scales) + n_channels)
        float32; the trailing n_channels block is the per-channel presence
        fraction over the FULL window (DL4.2 / W2.1: statistics come from
        present values only, and the presence fraction itself is passed as its
        own feature so the model is told how much it is trusting).

        Presence fraction is never NaN -- an all-absent channel has a presence
        fraction of exactly 0.0, which is a measurement, not a gap.
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
            # Causal: the last n_steps samples of the window.
            stats = self._slice_stats(X[:, :, -n_steps:], mask[:, :, -n_steps:])
            for c in range(n_channels):
                for stat in self.stats:
                    out[:, col] = stats[stat][:, c]
                    col += 1

        # Per-channel presence fraction over the FULL window.
        out[:, col : col + n_channels] = mask.astype(np.float64).mean(axis=-1)
        col += n_channels

        assert col == out.shape[1], f"emitted {col} columns, expected {out.shape[1]}"
        return out
