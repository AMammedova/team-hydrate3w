"""Member 2, W2.3 — hyperparameter search for XGBoostBaseline, validation
folds only. Record how many configurations were tried (paper needs this
number alongside Member 3's, per W2.3/W3.9 -- comparable tuning budget)."""


def search(X_train, mask_train, y_train, groups_train, param_grid: dict) -> dict:
    raise NotImplementedError
