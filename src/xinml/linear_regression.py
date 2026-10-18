"""Ordinary least-squares linear regression using NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert array-like input to finite, real floating-point values."""
    try:
        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError("Complex values are not supported.")
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_X(X: ArrayLike) -> NDArray[np.float64]:
    """Interpret 1D input as samples of a single feature."""
    X = _as_float_array(X, "X")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 1D single-feature array or a 2D feature matrix.")
    if 0 in X.shape:
        raise ValueError("X must contain at least one sample and one feature.")
    return X


def _validate_y(y: ArrayLike, n_samples: int) -> NDArray[np.float64]:
    """Require one target value per sample."""
    y = _as_float_array(y, "y")
    if y.ndim != 1:
        raise ValueError("y must be a 1D target array.")
    if y.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    return y


class LinearRegression:
    """Fit the single-target model ``y = X @ coef_ + intercept_``.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Include a constant term. If False, fix the intercept at zero.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or None
        Fitted feature coefficients; None before fitting.
    intercept_ : float or None
        Fitted intercept, or zero when disabled; None before fitting.
    n_features_in_ : int or None
        Number of features used during fitting; None before fitting.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be a boolean.")
        self.fit_intercept = fit_intercept
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegression:
        """Minimize ``||y - A @ beta||²`` and return this fitted model.

        X has shape (n_samples, n_features), or (n_samples,) for one
        feature. y must have shape (n_samples,). With an intercept, A
        prepends a column of ones to X. Otherwise A is simply X.

        ``lstsq`` solves least squares without forming an inverse. For a
        rank-deficient A, it returns the minimum-norm minimizing beta.
        """
        X = _validate_X(X)
        y = _validate_y(y, X.shape[0])
        design = np.column_stack((np.ones(X.shape[0]), X)) if self.fit_intercept else X
        solution, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

        # The first coefficient multiplies the constant column when present.
        if self.fit_intercept:
            self.intercept_ = float(solution[0])
            self.coef_ = solution[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = solution
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return one prediction per sample using the fitted linear model.

        Input follows the same shape rules as fit. For a single sample
        with several features, use a 2D row such as ``[[x1, x2]]``.
        Raises RuntimeError before fitting, or ValueError for invalid input.
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit(X, y) before predict or score.")
        X = _validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; received {X.shape[1]}."
            )
        return X @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R² = 1 - sum((y - prediction)²) / sum((y - mean(y))²).

        A perfect fit scores 1; predictions worse than the target mean
        can score below zero. Raises ValueError for fewer than two samples
        or a constant target, where R² is undefined. Other input and fitted
        state checks follow fit and predict.
        """
        predictions = self.predict(X)
        y = _validate_y(y, predictions.shape[0])
        if y.size < 2:
            raise ValueError("R-squared requires at least two samples.")
        if np.all(y == y[0]):
            raise ValueError("R-squared is undefined when y is constant.")
        residual_sum = np.sum((y - predictions) ** 2)
        total_sum = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - residual_sum / total_sum)
