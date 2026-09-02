"""Member 1, W1.4 — per-variable availability + frozen-value analysis.
Produces the paper table: variable, % present, % frozen, keep/drop.

Same question as tools/channel_availability.py (which channels are worth
keeping?), answered from already-loaded InstanceRecords instead of raw
parquet row-group statistics -- this is the version build_cache and the
report both call.
"""
from __future__ import annotations

import pandas as pd

from src.data.inventory import InstanceRecord
from src.data.windowing import frozen_run_mask


def variable_availability_table(
    instances: list[InstanceRecord],
    frozen_run_seconds: int = 60,
    max_missing_frac: float = 0.5,
) -> pd.DataFrame:
    """
    One row per raw variable seen in `instances`:

        variable               column name
        pct_missing            weighted mean raw NaN fraction
        pct_frozen             weighted mean frozen-run fraction (>= frozen_run_seconds
                                identical consecutive samples; disjoint from pct_missing,
                                since a NaN breaks a run rather than extending it -- see
                                windowing.frozen_run_mask)
        pct_effective_missing  pct_missing + pct_frozen -- what VariableSelector.fit()
                                actually thresholds against (it converts frozen runs to
                                NaN before computing missing_frac_)
        keep                   pct_effective_missing <= max_missing_frac

    Weighted by each instance's n_timesteps, so a 40-hour instance counts more
    than a 1-hour one -- same reasoning as tools/channel_availability.py's
    per-well weighting.

    Instances need not share the same columns (different wells instrument
    different channels): a variable's weighted mean only ever includes the
    instances that actually contain that column.
    """
    if not instances:
        raise ValueError("variable_availability_table() received no instances")

    variables = sorted({col for inst in instances for col in inst.variable_columns()})

    rows = []
    for var in variables:
        weight_total = 0.0
        missing_weighted = 0.0
        frozen_weighted = 0.0
        for inst in instances:
            if var not in inst.df.columns:
                continue
            n = inst.n_timesteps
            if n == 0:
                continue
            values = inst.df[var].to_numpy(dtype="float64")
            missing_frac = float(pd.isna(values).mean())
            frozen_frac = float(frozen_run_mask(values, frozen_run_seconds).mean())
            weight_total += n
            missing_weighted += missing_frac * n
            frozen_weighted += frozen_frac * n

        pct_missing = missing_weighted / weight_total if weight_total else 1.0
        pct_frozen = frozen_weighted / weight_total if weight_total else 0.0
        pct_effective_missing = pct_missing + pct_frozen
        rows.append(
            {
                "variable": var,
                "pct_missing": pct_missing,
                "pct_frozen": pct_frozen,
                "pct_effective_missing": pct_effective_missing,
                "keep": pct_effective_missing <= max_missing_frac,
            }
        )

    return pd.DataFrame(rows).sort_values("pct_effective_missing").reset_index(drop=True)
