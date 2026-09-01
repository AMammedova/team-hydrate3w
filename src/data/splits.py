"""
Module 3 — Grouped Cross-Validation Splitter.
See DL_Project_Statement_Hydrate3W.docx, section 6 (DL3.1-DL3.3).

IMPORTANT: split() yields a NESTED (train_idx, val_idx, test_idx), not a
two-way split. val_idx is carved out of that fold's training wells only
-- never from that fold's test wells -- because Module 7's early
stopping and Module 8's threshold/smoothing selection both need a
validation set that is never the test set.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class GroupedKFoldSplitter:
    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 3,
        val_frac: float = 0.2,
        group_col: str = "well_id",
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.val_frac = val_frac
        self.group_col = group_col
        self.random_state = random_state

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yields (train_idx, val_idx, test_idx) per (repeat, fold).

        Step 1: outer StratifiedGroupKFold on (y, groups) -> train_wells, test_wells.
        Step 2: inner grouped split restricted to train_wells only, carving
                out val_frac of those wells (by well, not by row) as val_idx.
        Simulated pseudo-group wells (see inventory.py) may end up in
        train_idx; they must never end up in val_idx or test_idx (DL3.2)
        -- filter them out of the outer split's candidate test/val pool
        before calling StratifiedGroupKFold if your run includes them.
        """
        # TODO: implement steps 1-2 above. sklearn's StratifiedGroupKFold
        # doesn't support repeats natively -- loop n_repeats times with a
        # different random_state derived per repeat (e.g.
        # self.random_state + repeat_idx) and reshuffle.
        raise NotImplementedError

    def fold_report(self, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
        """
        Per (repeat, fold): n_train_wells, n_val_wells, n_test_wells,
        n_val_positive_events, n_test_positive_events.

        Run this BEFORE any model training (DL3.3) -- it's Table 1 in
        the report and it's what tells you whether n_splits/n_repeats
        are sane given ~57 real positive instances. If any fold shows
        n_val_positive_events == 0, widen val_frac for that fold or
        merge it with an adjacent fold rather than proceeding.
        """
        # TODO
        raise NotImplementedError
