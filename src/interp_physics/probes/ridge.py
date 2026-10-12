"""Closed-form ridge regression for linear probes."""

from __future__ import annotations

import numpy as np


def _as_2d(y: np.ndarray) -> tuple[np.ndarray, bool]:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y[:, None], True
    return y, False


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Held-out R². A constant predictor that uses the eval-set mean scores 0."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    centered = y_true - y_true.mean(axis=0)
    ss_tot = np.sum(centered**2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Accuracy when every label is ±1. Otherwise the target is not a sign."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    labels = np.unique(np.round(y_true, decimals=6))
    if labels.size == 0 or not np.all(np.isin(labels, (-1.0, 1.0))):
        return None
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    l2: float = 1.0,
) -> dict[str, float | None]:
    """Fit ridge on train rows and score on eval rows.

    Features are centered and scaled with train statistics. The intercept is
    unpenalized. `l2` penalizes the weights in that standardized basis.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    y_train_2d, _ = _as_2d(y_train)
    y_eval_2d, _ = _as_2d(y_eval)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean = y_train_2d.mean(axis=0)

    xc = (x_train - x_mean) / x_std
    yc = y_train_2d - y_mean
    d = xc.shape[1]
    xtx = xc.T @ xc
    xtx.flat[:: d + 1] += l2
    weight = np.linalg.solve(xtx, xc.T @ yc)

    def predict(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std) @ weight + y_mean

    pred_eval = predict(x_eval)
    out: dict[str, float | None] = {
        "r2": r2_score(y_eval_2d, pred_eval),
        "accuracy": binary_accuracy(y_eval_2d, pred_eval),
    }
    return out


def readout_basis(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    l2: float = 1.0,
) -> np.ndarray:
    """Orthonormal columns in raw feature space that carry the linear readout.

    Shape ``[D, K]``. ``K`` is 1 for a scalar label and up to the number of
    outputs for a vector label. Empty if the fit has no direction.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train_2d, _ = _as_2d(y_train)
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean = y_train_2d.mean(axis=0)
    xc = (x_train - x_mean) / x_std
    yc = y_train_2d - y_mean
    d = xc.shape[1]
    xtx = xc.T @ xc
    xtx.flat[:: d + 1] += l2
    weight = np.linalg.solve(xtx, xc.T @ yc)
    raw = weight / x_std[:, None]
    singular = np.linalg.svd(raw, full_matrices=False, compute_uv=True)
    left, values, _right = singular
    if values.size == 0 or values[0] < 1e-10:
        return np.zeros((d, 0), dtype=np.float64)
    keep = values > 1e-6 * values[0]
    return left[:, keep]
