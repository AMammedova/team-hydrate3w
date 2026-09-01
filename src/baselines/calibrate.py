"""Member 2, W2.4 — isotonic regression or Platt scaling on validation
folds; reliability diagrams before/after. Feeds src/eval/metrics.py's
expected_calibration_error() and src/eval/plots.py's reliability figure."""


def fit_calibrator(y_val_true, y_val_prob, method: str = "isotonic"):
    raise NotImplementedError
