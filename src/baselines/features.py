"""
Module 4 (features) — see project statement section 7, and Addendum A.7
for the multi-timescale + mask-aware requirements.
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

    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        X, mask: (n_windows, n_channels, window_size) -- channels-first.
        Returns (n_windows, n_channels * len(stats) * len(scales) + n_channels)
        -- the extra n_channels block is per-channel presence fraction
        (DL4.2 / W2.1: computed over present values only, with the
        presence fraction itself passed as a feature).
        """
        # TODO: for each scale, slice the last `scale * window_size` steps;
        # for each channel, compute each stat in self.stats over the
        # PRESENT (mask==1) values only. slope = least-squares fit vs. the
        # time axis. last_diff = last present value minus first present
        # value in that slice. Concatenate across scales and stats, then
        # append the per-channel, full-window presence fraction
        # (mask.mean(axis=-1)) as its own feature block.
        raise NotImplementedError
