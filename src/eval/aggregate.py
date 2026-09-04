"""
Module 8 — results.csv -> paper tables. M5, W4.4.

Named `aggregate.py`, not `report.py`, on purpose -- this repo already
has a `report/` directory holding the actual paper (report.tex,
report.pdf); a file called report.py sitting in a different directory
doing something completely different was a genuine, avoidable source of
confusion and has been renamed everywhere it appeared.

Contract: results.csv columns are defined in src/contract.RESULTS_COLUMNS:
    model, fold, seed, condition, metric_name, value
No number in the paper is ever typed by hand -- everything comes from
results/ via this module (team contract §0.3).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.contract import RESULTS_COLUMNS


def load_results(path: str = "results/results.csv") -> pd.DataFrame:
    """Load the cross-validation results CSV.

    Validates that the required columns exist per the contract.
    """
    df = pd.read_csv(path)
    missing = set(RESULTS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {missing}. "
            f"Expected columns: {RESULTS_COLUMNS}"
        )
    return df


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


def to_latex_table(
    summary: pd.DataFrame,
    metric_name: str,
    out_path: str,
    caption: str | None = None,
    label: str | None = None,
) -> None:
    """Write one metric's summary as a LaTeX table for report.tex to
    \\input directly -- no hand-typed numbers (W4.4).

    Parameters
    ----------
    summary : pd.DataFrame
        Output of summarize_folds().
    metric_name : str
        Which metric to extract (e.g. 'pr_auc', 'event_recall', 'far').
    out_path : str
        File path to write the .tex table snippet.
    caption : str, optional
        LaTeX table caption.
    label : str, optional
        LaTeX table label for cross-referencing.
    """
    df = summary[summary["metric_name"] == metric_name].copy()
    if df.empty:
        return

    # Format as "mean ± std" with appropriate precision
    def _fmt(row: pd.Series) -> str:
        mean_val = row["mean"]
        std_val = row["std"]
        if abs(mean_val) < 1:
            return f"${mean_val:.3f} \\pm {std_val:.3f}$"
        elif abs(mean_val) < 100:
            return f"${mean_val:.2f} \\pm {std_val:.2f}$"
        else:
            return f"${mean_val:.1f} \\pm {std_val:.1f}$"

    df["value_str"] = df.apply(_fmt, axis=1)

    # Pivot: rows = model, columns = condition
    pivot_df = df.pivot(index="model", columns="condition", values="value_str")

    # Build LaTeX table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    n_cols = len(pivot_df.columns)
    col_spec = "l" + "c" * n_cols
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    # Header
    header = "Model & " + " & ".join(str(c) for c in pivot_df.columns) + " \\\\"
    lines.append(header)
    lines.append("\\hline")

    # Data rows
    for model_name, row in pivot_df.iterrows():
        row_str = str(model_name) + " & " + " & ".join(str(row[c]) for c in pivot_df.columns) + " \\\\"
        lines.append(row_str)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)


def generate_all_tables(
    results_path: str = "results/results.csv",
    tables_dir: str = "report/tables/",
) -> None:
    """Generate all LaTeX tables from results.csv.

    Called by run_all.sh as the final aggregation step.
    """
    df = load_results(results_path)
    summary = summarize_folds(df)

    # List of (metric_name, filename, caption, label)
    table_specs = [
        ("pr_auc", "pr_auc.tex",
         "PR-AUC (mean $\\pm$ std across folds)", "tab:pr_auc"),
        ("event_recall", "event_recall.tex",
         "Event Recall at matched FAR budget", "tab:event_recall"),
        ("far", "far.tex",
         "False Alarm Rate (per operating hour)", "tab:far"),
        ("lead_time", "lead_time.tex",
         "Median Lead Time (seconds) before transient onset", "tab:lead_time"),
        ("ece", "ece.tex",
         "Expected Calibration Error", "tab:ece"),
    ]

    for metric_name, filename, caption, label in table_specs:
        if metric_name in summary["metric_name"].values:
            out_path = os.path.join(tables_dir, filename)
            to_latex_table(summary, metric_name, out_path, caption=caption, label=label)
