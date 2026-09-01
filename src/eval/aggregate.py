"""
Module 8 — results.csv -> paper tables. Member 4, W4.4.

Named `aggregate.py`, not `report.py`, on purpose -- this repo already
has a `report/` directory holding the actual paper (report.tex,
report.pdf); a file called report.py sitting in a different directory
doing something completely different was a genuine, avoidable source of
confusion and has been renamed everywhere it appeared.
"""

from __future__ import annotations

import pandas as pd


def load_results(path: str = "results/results.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    fold_metrics: results.csv loaded via load_results() -- columns
    model, fold, seed, condition, metric_name, value.
    Returns one row per (model, condition, metric_name) with mean and std
    across fold x seed.
    """
    return (
        fold_metrics
        .groupby(["model", "condition", "metric_name"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def to_latex_table(summary: pd.DataFrame, metric_name: str, out_path: str) -> None:
    """Write one metric's summary as a LaTeX table into report/tables/,
    for report.tex to \\input directly -- no hand-typed numbers (W4.4)."""
    # TODO: filter summary to metric_name, pivot model x condition,
    # format as "mean ± std", write a minimal tabular environment.
    raise NotImplementedError
