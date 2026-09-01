"""Member 1, W1.4 — per-variable availability + frozen-value analysis.
Produces the paper table: variable, % present, % frozen, keep/drop."""
from __future__ import annotations
import pandas as pd


def variable_availability_table(instances: list) -> pd.DataFrame:
    # TODO: for each variable, aggregate InstanceRecord.missing_fraction()
    # across `instances`; separately flag frozen-run rate (see
    # VariableSelector.frozen_run_seconds in windowing.py) per variable.
    raise NotImplementedError
