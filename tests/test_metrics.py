"""
Tests for Module 8 — metrics.py
"""

from __future__ import annotations

import numpy as np
import math

from src.eval.metrics import (
    positive_score,
    pr_auc,
    pr_auc_per_class,
    event_recall,
    false_alarms_per_operating_hour,
    expected_calibration_error,
)
from src.contract import NORMAL, TRANSIENT, ESTABLISHED


def test_positive_score():
    class_probs = np.array([
        [0.8, 0.1, 0.1],  # Mostly normal
        [0.2, 0.7, 0.1],  # Mostly transient
        [0.1, 0.1, 0.8],  # Mostly established
    ])
    score = positive_score(class_probs)
    np.testing.assert_allclose(score, [0.2, 0.8, 0.9])


def test_pr_auc():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    auc = pr_auc(y_true, y_prob)
    assert 0.0 <= auc <= 1.0


def test_pr_auc_per_class():
    y_true = np.array([NORMAL, NORMAL, TRANSIENT, ESTABLISHED])
    class_probs = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
    ])
    aucs = pr_auc_per_class(y_true, class_probs)
    assert "normal" in aucs
    assert "transient" in aucs
    assert "established" in aucs
    assert aucs["normal"] > 0.5


def test_event_recall():
    events_flagged = [True, False, True]
    recall = event_recall(events_flagged, 3)
    assert math.isclose(recall, 2/3)
    
    assert math.isnan(event_recall([], 0))


def test_false_alarms_per_operating_hour():
    far = false_alarms_per_operating_hour(5, 10.0)
    assert math.isclose(far, 0.5)
    
    assert math.isnan(false_alarms_per_operating_hour(5, 0.0))


def test_expected_calibration_error():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Perfectly calibrated
    y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert math.isclose(ece, 0.0)
    
    # Poorly calibrated
    y_prob_poor = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    ece_poor = expected_calibration_error(y_true, y_prob_poor, n_bins=10)
    assert math.isclose(ece_poor, 1.0)
