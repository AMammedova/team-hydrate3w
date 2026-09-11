"""
Unit tests for src/data/fold_report_latex.py (Module 3, Member 2).

report.tex \input{}s this file's output, so what these guard is not
appearance but arithmetic reaching the paper: every fold present, every
Normal-hour figure carried across, and a header saying the file is
generated so nobody edits the numbers in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fold_report_latex import fold_report_to_latex
from src.data.splits import GroupedKFoldSplitter

# The fixture lives in test_splits.py: these tests render what fold_report()
# actually produces, not a hand-built DataFrame that could drift from it.
from test_splits import make_dataset


def test_fold_report_latex_carries_every_fold_and_the_hour_columns():
    r"""
    report.tex \input{}s this file, so a silently truncated table would put
    a wrong fold count in the paper rather than raising anything.
    """
    y, groups, instances, is_sim, hours = make_dataset()
    splitter = GroupedKFoldSplitter(n_splits=3)
    report = splitter.fold_report(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    )
    tex = fold_report_to_latex(report, label="tab:folds")

    assert tex.count(chr(92) + chr(92) + "\n") >= len(report)   # one row each
    for _, r in report.iterrows():
        assert f"{r['test_normal_hours']:.1f}" in tex
        assert f"{r['val_normal_hours']:.1f}" in tex
    assert "tab:folds" in tex
    assert tex.count("begin{table}") == 1 and tex.count("end{table}") == 1
    assert "toprule" in tex and "bottomrule" in tex


def test_fold_report_latex_is_regenerable_not_hand_edited():
    """The header must say so -- the file lives in the report tree."""
    y, groups, instances, is_sim, hours = make_dataset()
    report = GroupedKFoldSplitter(n_splits=3).fold_report(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    )
    tex = fold_report_to_latex(report)
    assert "Auto-generated" in tex
    assert "src.data.splits" in tex
