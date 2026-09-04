"""
Module 8 — metric implementations. Member 4, W4.1.

positive_score() is the single, pre-registered 3-class -> positive-score
reduction (default: P(Transient) + P(Established)). alarm.py and
thresholds.py both call this rather than re-deriving their own reduction
-- see the project statement's "Pre-registered decision" callout for why
P(Transient)-only is rejected as the default.

This file holds metric FUNCTIONS only (given predictions and ground
truth, return a number). Alarm timing lives in alarm.py; threshold
selection lives in thresholds.py; turning results.csv into tables lives
in aggregate.py; figures live in plots.py. Keeping these separate is
deliberate -- see the note in the repo README about not overloading one
file with everything "evaluation-shaped".
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

# Column order convention used throughout: [Normal, Transient, Established]
from src.contract import NORMAL, TRANSIENT, ESTABLISHED


def positive_score(class_probs: np.ndarray) -> np.ndarray:
    """
    class_probs: (n_windows, 3) softmax output.
    Pre-registered default: P(Transient) + P(Established), i.e. 1 - P(Normal).
    """
    return class_probs[:, TRANSIENT] + class_probs[:, ESTABLISHED]


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """y_true: binary (positive = Transient or Established). y_prob: from positive_score()."""
    return float(average_precision_score(y_true, y_prob))


def pr_auc_per_class(y_true_multiclass: np.ndarray, class_probs: np.ndarray) -> dict:
    """One-vs-rest PR-AUC for each of Normal/Transient/Established -- a richer
    diagnostic alongside the collapsed positive_score() used for alarms."""
    out = {}
    for name, idx in (("normal", NORMAL), ("transient", TRANSIENT), ("established", ESTABLISHED)):
        y_bin = (y_true_multiclass == idx).astype(int)
        out[name] = float(average_precision_score(y_bin, class_probs[:, idx]))
    return out


def event_recall(events_flagged: list, events_total: int) -> float:
    """events_flagged: list of bool, one per real hydrate event, True if
    alarm.alarm_times() fired at least once before that event's failure_time."""
    if events_total == 0:
        return float("nan")
    return sum(events_flagged) / events_total


def false_alarms_per_operating_hour(n_false_alarm_onsets: int, total_normal_hours: float) -> float:
    """Count alarm ONSETS on Normal-operation data, not samples/windows --
    one long false alarm must not count as thousands (W4.1 pitfall)."""
    if total_normal_hours <= 0:
        return float("nan")
    return n_false_alarm_onsets / total_normal_hours


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE: |accuracy - confidence| per bin, weighted by bin size."""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (binids == i)
        if not np.any(bin_mask):
            continue
        
        bin_acc = np.mean(y_true[bin_mask])
        bin_conf = np.mean(y_prob[bin_mask])
        bin_weight = np.sum(bin_mask) / len(y_prob)
        
        ece += bin_weight * np.abs(bin_acc - bin_conf)
        
    return float(ece)
