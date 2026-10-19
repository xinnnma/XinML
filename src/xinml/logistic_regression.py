"""Binary logistic regression trained with NumPy batch gradient descent."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linear_regression import _validate_X, _validate_y


def _sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute 1 / (1 + exp(-z)) without exponentiating positive values."""
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    # Separate branches avoid overflow for large positive or negative logits.
    # Underflow to zero is the correct limiting behavior in the tails.
    with np.errstate(under="ignore"):
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    return probabilities


def _validate_binary_y(y: ArrayLike, n_samples: int) -> NDArray[np.float64]:
    """Reuse the regression shape checks and require labels in {0, 1}."""
    y = _validate_y(y, n_samples)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only binary labels 0 and 1.")
    return y


class LogisticRegression:
    """Model P(y=1 | X) as sigmoid(X @ coef_ + intercept_).

    Parameters
    ----------
    learning_rate : float, default=0.1
        Positive, finite step size for batch gradient descent.
    max_iter : int, default=1000
        Positive number of gradient updates. No early stopping is applied.
    fit_intercept : bool, default=True
        Include a constant term. If False, fix the intercept at zero.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or None
        Fitted feature coefficients; None before fitting.
    intercept_ : float or None
        Fitted intercept, or zero when disabled; None before fitting.
    n_features_in_ : int or None
        Number of training features; None before fitting.
    loss_history_ : list of float
        Mean binary cross-entropy at initialization and after each update.
        Empty before fitting; length max_iter + 1 after a successful fit.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        fit_intercept: bool = True,
    ) -> None:
        if (
            isinstance(learning_rate, (bool, np.bool_))
            or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number.")
        if (
            isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, Integral)
            or max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be a boolean.")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression:
        """Minimize mean binary cross-entropy and return this fitted model.

        X has shape (n_samples, n_features), or (n_samples,) for one
        feature. y has shape (n_samples,) and must contain both 0 and 1.

        With design matrix A and logits z = A @ weights, the loss is
        mean(log(1 + exp(z)) - y*z), and its gradient is A.T @ (p-y) / n.
        Each update subtracts learning_rate times this gradient. A column
        of ones learns the optional intercept. Every fit starts from zero.
        """
        X = _validate_X(X)
        y = _validate_binary_y(y, X.shape[0])
        if np.unique(y).size != 2:
            raise ValueError("y must contain both binary classes 0 and 1 for training.")
        design = np.column_stack((np.ones(X.shape[0]), X)) if self.fit_intercept else X
        weights = np.zeros(design.shape[1])
        logits = np.zeros(X.shape[0])
        losses = [float(np.log(2.0))]

        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                for _ in range(self.max_iter):
                    errors = _sigmoid(logits) - y
                    # Average before the matrix product to avoid an oversized sum.
                    gradient = design.T @ (errors / X.shape[0])
                    weights = weights - self.learning_rate * gradient
                    logits = design @ weights
                    # For binary y, this equals the loss above without log(0)
                    # or cancellation of large, nearly equal terms.
                    loss = np.mean(np.logaddexp(0.0, (1.0 - 2.0 * y) * logits))
                    losses.append(float(loss))
        except FloatingPointError as exc:
            raise ValueError(
                "Training overflowed. Scale X or reduce learning_rate."
            ) from exc

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        self.n_features_in_ = X.shape[1]
        self.loss_history_ = losses
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return shape (n_samples, 2), with columns P(y=0) and P(y=1).

        Input follows fit's shape rules. Use [[x1, x2]] for one sample
        with two features. Raises RuntimeError before fitting, or
        ValueError for invalid input, feature counts, or numeric overflow.
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit(X, y) before predict_proba, predict, or score.")
        X = _validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; received {X.shape[1]}."
            )
        try:
            with np.errstate(over="raise", invalid="raise"):
                logits = X @ self.coef_ + self.intercept_
        except FloatingPointError as exc:
            raise ValueError("Prediction overflowed. Scale X before predicting.") from exc
        probabilities = _sigmoid(logits)
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """Return 0 or 1 per sample; probabilities at least 0.5 map to 1."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the fraction of correct predictions for binary labels.

        An evaluation set may contain just one of the two classes.
        Input and fitted-state checks follow fit and predict_proba.
        """
        predictions = self.predict(X)
        y = _validate_binary_y(y, predictions.shape[0])
        return float(np.mean(predictions == y))
