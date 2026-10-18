"""Check linear regression against analytical solutions and invalid inputs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from xinml import LinearRegression


@pytest.fixture
def fitted_model():
    """A known line: y = 2x + 1."""
    return LinearRegression().fit([0, 1, 2, 3], [1, 3, 5, 7])


def test_known_single_feature_coefficients():
    model = LinearRegression()
    result = model.fit([0, 1, 2, 3], [1, 3, 5, 7])

    assert result is model
    assert model.coef_.shape == (1,)
    assert_allclose(model.coef_, [2.0], rtol=1e-10, atol=1e-10)
    assert model.intercept_ == pytest.approx(1.0, abs=1e-10)
    assert model.n_features_in_ == 1


@pytest.mark.parametrize("X", [[4, 5], [[4], [5]]], ids=["1d", "2d"])
def test_predictions_on_unseen_samples(fitted_model, X):
    predictions = fitted_model.predict(X)
    assert predictions.shape == (2,)
    assert_allclose(predictions, [9.0, 11.0], rtol=1e-10, atol=1e-10)


def test_multiple_features():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, -1], [3, 2]])
    y = 4 + 2 * X[:, 0] - 3 * X[:, 1]
    model = LinearRegression().fit(X, y)

    assert model.n_features_in_ == 2
    assert_allclose(model.coef_, [2.0, -3.0], rtol=1e-10, atol=1e-10)
    assert model.intercept_ == pytest.approx(4.0, abs=1e-10)
    assert_allclose(model.predict([[2, 3]]), [-1.0], rtol=1e-10, atol=1e-10)
    assert_allclose(model.predict([[2, 3], [4, -1]]), [-1.0, 15.0], atol=1e-10)


@pytest.mark.parametrize(
    "fit_intercept, expected_slope, expected_intercept",
    [(True, 2.0, 3.0), (False, 23.0 / 7.0, 0.0)],
)
def test_optional_intercept(fit_intercept, expected_slope, expected_intercept):
    # With no intercept, the optimal slope is sum(x*y) / sum(x*x) = 46/14.
    model = LinearRegression(fit_intercept=fit_intercept).fit([1, 2, 3], [5, 7, 9])
    assert_allclose(model.coef_, [expected_slope], rtol=1e-10, atol=1e-10)
    assert model.intercept_ == pytest.approx(expected_intercept, abs=1e-10)
    assert_allclose(
        model.predict([0, 4]),
        [expected_intercept, 4 * expected_slope + expected_intercept],
        rtol=1e-10,
        atol=1e-10,
    )


def test_least_squares_with_noisy_targets():
    # For centered x, slope is 1 and the intercept is mean(y) = 4/3.
    model = LinearRegression().fit([-1, 0, 1], [0, 2, 2])
    assert_allclose(model.coef_, [1.0], rtol=1e-10, atol=1e-10)
    assert model.intercept_ == pytest.approx(4.0 / 3.0, abs=1e-10)
    assert_allclose(model.predict([-1, 0, 1]), [1 / 3, 4 / 3, 7 / 3], atol=1e-10)


def test_r_squared_perfect_fit(fitted_model):
    assert fitted_model.score([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx(1.0)


def test_r_squared_known_residuals(fitted_model):
    # Predictions [1, 3, 5] give residual sum 2 and total sum 14.
    assert fitted_model.score([0, 1, 2], [1, 2, 6]) == pytest.approx(6.0 / 7.0)


def test_r_squared_can_be_negative(fitted_model):
    assert fitted_model.score([0, 1, 2], [-10, -9, -8]) == pytest.approx(-216.0)


@pytest.mark.parametrize(
    "X, y, message",
    [
        (1.0, [1], "X must be a 1D"),
        (np.ones((2, 1, 1)), [1, 2], "X must be a 1D"),
        ([1, 2], 1.0, "y must be a 1D"),
        ([1, 2], [[1], [2]], "y must be a 1D"),
        ([1, 2], np.ones((2, 1, 1)), "y must be a 1D"),
        ([1, 2], [1], "same number of samples"),
        ([1, 2], [], "same number of samples"),
        ([], [], "at least one sample and one feature"),
        (np.empty((0, 2)), [], "at least one sample and one feature"),
        (np.empty((2, 0)), [1, 2], "at least one sample and one feature"),
    ],
)
def test_fit_rejects_invalid_dimensions(X, y, message):
    with pytest.raises(ValueError, match=message):
        LinearRegression().fit(X, y)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["X", "y"])
def test_fit_rejects_nonfinite_values(value, field):
    X, y = [0.0, 1.0], [1.0, 3.0]
    if field == "X":
        X[0] = value
    else:
        y[0] = value
    with pytest.raises(ValueError, match=f"{field} must contain only finite values"):
        LinearRegression().fit(X, y)


@pytest.mark.parametrize(
    "X, y, field",
    [
        (["bad", "data"], [1, 2], "X"),
        ([0, 1], ["bad", "data"], "y"),
        ([1 + 2j, 2], [1, 2], "X"),
        ([0, 1], [1 + 2j, 2], "y"),
        ([[1], [2, 3]], [1, 2], "X"),
    ],
)
def test_fit_rejects_nonnumeric_complex_or_ragged_input(X, y, field):
    with pytest.raises(ValueError, match=f"{field} must contain real numeric values"):
        LinearRegression().fit(X, y)


@pytest.mark.parametrize("fit_intercept", [None, "False", 1])
def test_intercept_option_must_be_boolean(fit_intercept):
    with pytest.raises(ValueError, match="fit_intercept must be a boolean"):
        LinearRegression(fit_intercept=fit_intercept)


@pytest.mark.parametrize("method", ["predict", "score"])
def test_methods_require_fit(method):
    model = LinearRegression()
    arguments = ([1, 2],) if method == "predict" else ([1, 2], [1, 2])
    with pytest.raises(RuntimeError, match="Call fit"):
        getattr(model, method)(*arguments)


@pytest.mark.parametrize(
    "X, message",
    [
        (1.0, "X must be a 1D"),
        (np.ones((2, 1, 1)), "X must be a 1D"),
        ([], "at least one sample and one feature"),
        ([[1, 2]], "X must have 1 features"),
        ([np.nan, 1], "X must contain only finite values"),
        ([np.inf, 1], "X must contain only finite values"),
        (["bad"], "X must contain real numeric values"),
    ],
)
def test_predict_rejects_invalid_input(fitted_model, X, message):
    with pytest.raises(ValueError, match=message):
        fitted_model.predict(X)


@pytest.mark.parametrize(
    "X, y, message",
    [
        ([0], [1], "at least two samples"),
        ([0, 1], [3, 3], "y is constant"),
        ([0, 1], [[1], [3]], "y must be a 1D"),
        ([0, 1], [1], "same number of samples"),
        ([0, 1], [1, np.nan], "y must contain only finite values"),
        ([0, 1], [1, np.inf], "y must contain only finite values"),
        ([0, 1], ["bad", "data"], "y must contain real numeric values"),
        ([[0, 1], [1, 2]], [1, 3], "X must have 1 features"),
    ],
)
def test_score_rejects_invalid_input(fitted_model, X, y, message):
    with pytest.raises(ValueError, match=message):
        fitted_model.score(X, y)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_rank_deficient_features(fit_intercept):
    X = np.array([[0, 0], [1, 2], [2, 4], [3, 6]], dtype=float)
    intercept = 1.0 if fit_intercept else 0.0
    y = 3 * X[:, 0] + intercept
    model = LinearRegression(fit_intercept=fit_intercept).fit(X, y)

    # Among coefficients satisfying b1 + 2*b2 = 3, [0.6, 1.2] has least norm.
    assert_allclose(model.coef_, [0.6, 1.2], rtol=1e-10, atol=1e-10)
    assert model.intercept_ == pytest.approx(intercept, abs=1e-10)
    assert_allclose(model.predict(X), y, rtol=1e-10, atol=1e-10)
    assert_allclose(model.predict([[4, 8]]), [12 + intercept], atol=1e-10)


def test_constant_feature_redundant_with_intercept():
    model = LinearRegression().fit([[1], [1], [1]], [2, 3, 4])
    assert np.all(np.isfinite(model.coef_))
    assert_allclose(model.predict([[1], [1]]), [3.0, 3.0], atol=1e-10)


def test_underdetermined_design():
    X = [[1, 0, 1], [0, 1, 1]]
    model = LinearRegression(fit_intercept=False).fit(X, [1, 2])
    assert_allclose(model.coef_, [0.0, 1.0, 1.0], atol=1e-10)
    assert_allclose(model.predict(X), [1.0, 2.0], atol=1e-10)


def test_refit_updates_feature_count(fitted_model):
    fitted_model.fit([[0, 0], [1, 0], [0, 1]], [4, 6, 1])
    assert fitted_model.n_features_in_ == 2
    assert_allclose(fitted_model.predict([[2, 3]]), [-1.0], atol=1e-10)


def test_fit_and_predict_do_not_modify_inputs():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])
    original_X, original_y = X.copy(), y.copy()
    model = LinearRegression().fit(X, y)
    model.predict(X)
    assert_allclose(X, original_X, atol=1e-10)
    assert_allclose(y, original_y, atol=1e-10)
