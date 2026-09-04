"""
Module 4 (features) — see project statement section 7, and Addendum A.7
for the multi-timescale + mask-aware requirements.

Multi-timescale rolling features over causal windows:
  - 3 time scales: full window, last half, last quarter
  - 6 statistics per scale per channel: mean, std, min, max, slope, last_diff
  - 1 presence fraction per channel (full window)
  - All computed over PRESENT (mask==1) values only
"""

from __future__ import annotations

import numpy as np


class RollingFeatureExtractor:
    def __init__(
        self,
        stats: list[str] | None = None,
        scales: list[float] | None = None,   # fraction of the window, from the end
    ) -> None:
        self.stats = stats or ["mean", "std", "min", "max", "slope", "last_diff"]
        # full window, last half, last quarter -- per Addendum A.7 / team W2.1
        self.scales = scales or [1.0, 0.5, 0.25]

    def _stat(self, values: np.ndarray, stat: str) -> float:
        """Compute one statistic over present (non-NaN) values in a 1-D slice."""
        if len(values) == 0:
            return 0.0
        if stat == "mean":
            return float(np.mean(values))
        if stat == "std":
            return float(np.std(values)) if len(values) > 1 else 0.0
        if stat == "min":
            return float(np.min(values))
        if stat == "max":
            return float(np.max(values))
        if stat == "slope":
            if len(values) < 2:
                return 0.0
            # Least-squares slope vs. equally-spaced time axis
            t = np.arange(len(values), dtype=np.float64)
            t -= t.mean()
            denom = np.dot(t, t)
            if denom == 0.0:
                return 0.0
            return float(np.dot(t, values.astype(np.float64)) / denom)
        if stat == "last_diff":
            return float(values[-1] - values[0]) if len(values) >= 2 else 0.0
        raise ValueError(f"Unknown stat: {stat!r}")

    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        X, mask: (n_windows, n_channels, window_size) -- channels-first.
        Returns (n_windows, n_channels * len(stats) * len(scales) + n_channels)
        -- the extra n_channels block is per-channel presence fraction
        (DL4.2 / W2.1: computed over present values only, with the
        presence fraction itself passed as a feature).
        """
        n_windows, n_channels, window_size = X.shape
        n_stats = len(self.stats)
        n_scales = len(self.scales)
        n_features = n_channels * n_stats * n_scales + n_channels

        out = np.zeros((n_windows, n_features), dtype=np.float32)

        for w in range(n_windows):
            col = 0
            # Scale features
            for scale in self.scales:
                n_steps = max(1, int(round(scale * window_size)))
                # Take last n_steps samples (causal: from the end)
                X_slice = X[w, :, -n_steps:]        # (n_channels, n_steps)
                mask_slice = mask[w, :, -n_steps:]  # (n_channels, n_steps)

                for c in range(n_channels):
                    present = X_slice[c][mask_slice[c].astype(bool)]
                    for stat in self.stats:
                        out[w, col] = self._stat(present, stat)
                        col += 1

            # Per-channel presence fraction over FULL window
            for c in range(n_channels):
                out[w, col] = float(mask[w, c].mean())
                col += 1

        return out
