"""
Unit tests for src/data/availability.py (Module 1, W1.4).

Builds tiny synthetic InstanceRecords with hand-computable missing/frozen
fractions, so the weighted-average and keep/drop logic can be checked
against numbers worked out by hand rather than just "it runs".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.availability import variable_availability_table
from src.data.inventory import InstanceRecord


def _make_record(instance_id: str, well_id: str, n: int, columns: dict[str, np.ndarray]) -> InstanceRecord:
    idx = pd.date_range("2020-01-01", periods=n, freq="1s")
    df = pd.DataFrame(columns, index=idx)
    df["class"] = 0
    return InstanceRecord(
        instance_id=instance_id,
        well_id=well_id,
        source="real",
        event_code=9,
        filepath=Path(f"{instance_id}.parquet"),
        df=df,
    )


def test_pct_missing_and_frozen_hand_computed():
    n = 100
    # "clean": no NaN, no frozen run.
    clean = np.arange(n, dtype="float64")
    # "half_missing": first half NaN.
    half_missing = np.arange(n, dtype="float64")
    half_missing[: n // 2] = np.nan
    # "frozen_tail": last 70 samples identical (frozen run >= 60s), rest varies.
    frozen_tail = np.arange(n, dtype="float64")
    frozen_tail[30:] = 5.0

    inst = _make_record(
        "WELL-00001_20200101000000", "WELL-00001", n,
        {"clean": clean, "half_missing": half_missing, "frozen_tail": frozen_tail},
    )

    table = variable_availability_table([inst], frozen_run_seconds=60, max_missing_frac=0.5)
    by_var = table.set_index("variable")

    assert by_var.loc["clean", "pct_missing"] == pytest.approx(0.0)
    assert by_var.loc["clean", "pct_frozen"] == pytest.approx(0.0)
    assert by_var.loc["clean", "keep"] == True  # noqa: E712

    assert by_var.loc["half_missing", "pct_missing"] == pytest.approx(0.5)
    assert by_var.loc["half_missing", "keep"] == True  # noqa: E712 (0.5 <= 0.5)

    assert by_var.loc["frozen_tail", "pct_frozen"] == pytest.approx(0.70)
    assert by_var.loc["frozen_tail", "pct_effective_missing"] == pytest.approx(0.70)
    assert by_var.loc["frozen_tail", "keep"] == False  # noqa: E712


def test_weighted_by_instance_length():
    # A short instance that is 100% missing should barely move the average
    # once a much longer clean instance is added.
    short_missing = _make_record(
        "WELL-00001_20200101000000", "WELL-00001", 10,
        {"chan": np.full(10, np.nan)},
    )
    long_clean = _make_record(
        "WELL-00002_20200102000000", "WELL-00002", 990,
        {"chan": np.arange(990, dtype="float64")},
    )
    table = variable_availability_table([short_missing, long_clean])
    pct_missing = table.set_index("variable").loc["chan", "pct_missing"]
    assert pct_missing == pytest.approx(10 / 1000)


def test_variable_not_present_in_every_instance_is_excluded_from_its_average():
    only_in_one = _make_record(
        "WELL-00001_20200101000000", "WELL-00001", 100,
        {"shared": np.zeros(100), "only_here": np.zeros(100)},
    )
    no_only_here = _make_record(
        "WELL-00002_20200102000000", "WELL-00002", 100,
        {"shared": np.zeros(100)},
    )
    table = variable_availability_table([only_in_one, no_only_here])
    assert set(table["variable"]) == {"shared", "only_here"}
    # only_here's average is computed over its one instance only (0% missing),
    # not diluted by the instance that never had the column.
    assert table.set_index("variable").loc["only_here", "pct_missing"] == pytest.approx(0.0)


def test_empty_instances_raises():
    with pytest.raises(ValueError):
        variable_availability_table([])
