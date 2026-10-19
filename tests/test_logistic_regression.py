"""Check binary logistic regression, optimization, and input validation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from xinml import LogisticRegression


@pytest.fixture
def fitted_model():
    return LogisticRegression(max_iter=300).fit([-3, -2, -1, 1, 2, 3], [0, 0, 0, 1, 1, 1])


def test_training_on_simple_binary_data():
    X, y = [-3, -2, -1, 1, 2, 3], [0, 0, 0, 1, 1, 1]
    model = LogisticRegression()
    result = model.fit(X, y)

    assert result is model
    assert model.coef_.shape == (1,)
    assert model.coef_[0] > 0
    assert model.intercept_ == pytest.approx(0.0, abs=1e-10)
    assert model.n_features_in_ == 1
    assert_allclose(model.predict(X), y, atol=1e-10)
    assert model.score(X, y) == pytest.approx(1.0)


@pytest.mark.parametrize("X", [[-2, 2], [[-2], [2]]], ids=["1d", "2d"])
def test_prediction_shapes_and_labels(fitted_model, X):
    predictions = fitted_model.predict(X)
    assert predictions.shape == (2,)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert np.all(np.isin(predictions, [0, 1]))
    assert_allclose(predictions, [0, 1], atol=1e-10)


@pytest.mark.parametrize("X", [[-2, 0, 2], [[-2], [0], [2]]], ids=["1d", "2d"])
def test_probability_shape_range_and_order(fitted_model, X):
    probabilities = fitted_model.predict_proba(X)
    assert probabilities.shape == (3, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert_allclose(probabilities.sum(axis=1), np.ones(3), atol=1e-12)
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[2, 1] > probabilities[2, 0]
    assert_allclose(probabilities[1], [0.5, 0.5], atol=1e-10)


def test_probability_ties_predict_class_one():
    model = LogisticRegression(max_iter=5).fit([0, 0], [0, 1])
    assert_allclose(model.predict_proba([0]), [[0.5, 0.5]], atol=1e-12)
    assert_allclose(model.predict([0]), [1], atol=1e-12)


def test_accuracy_on_unseen_samples(fitted_model):
    assert fitted_model.score([-4, -0.5, 0.5, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert fitted_model.score([-1, 1], [0, 0]) == pytest.approx(0.5)


def test_score_allows_one_class_in_evaluation(fitted_model):
    assert fitted_model.score([1, 2], [1, 1]) == pytest.approx(1.0)


def test_multiple_features():
    X = [[-2, -1], [-1, -2], [-1, 0], [0, -1], [1, 0], [0, 1], [1, 2], [2, 1]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    model = LogisticRegression().fit(X, y)

    assert model.n_features_in_ == 2
    assert model.coef_.shape == (2,)
    assert np.all(model.coef_ > 0)
    assert model.predict_proba([[1, 2]]).shape == (1, 2)
    assert_allclose(model.predict([[-2, 0], [0, 2]]), [0, 1], atol=1e-10)
    assert model.score(X, y) == pytest.approx(1.0)


@pytest.mark.parametrize("fit_intercept, expected_accuracy", [(True, 1.0), (False, 0.5)])
def test_optional_intercept(fit_intercept, expected_accuracy):
    # An origin-only boundary cannot separate two classes with all positive x.
    X, y = [1, 2, 3, 4], [0, 0, 1, 1]
    model = LogisticRegression(fit_intercept=fit_intercept, max_iter=2000).fit(X, y)
    assert model.score(X, y) == pytest.approx(expected_accuracy)
    if fit_intercept:
        assert model.intercept_ < 0
    else:
        assert model.intercept_ == pytest.approx(0.0, abs=1e-12)
        assert_allclose(model.predict_proba([0]), [[0.5, 0.5]], atol=1e-12)


@pytest.mark.parametrize("learning_rate, expected_slope", [(0.2, 0.1), (1.0, 0.5)])
def test_one_gradient_step_has_known_solution(learning_rate, expected_slope):
    # At zero weights, probabilities are 0.5 and the mean slope gradient is -0.5.
    model = LogisticRegression(learning_rate=learning_rate, max_iter=1).fit([-1, 1], [0, 1])
    assert_allclose(model.coef_, [expected_slope], atol=1e-12)
    assert model.intercept_ == pytest.approx(0.0, abs=1e-12)
    assert len(model.loss_history_) == 2


def test_loss_decreases_with_gradient_descent(fitted_model):
    losses = np.array(fitted_model.loss_history_)
    assert losses.shape == (301,)
    assert np.all(np.isfinite(losses))
    assert losses[0] == pytest.approx(np.log(2.0))
    assert np.all(np.diff(losses) <= 1e-12)
    assert losses[-1] < 0.1


def test_more_iterations_reduce_loss():
    X, y = [-2, -1, 1, 2], [0, 0, 1, 1]
    short = LogisticRegression(max_iter=1).fit(X, y)
    longer = LogisticRegression(max_iter=200).fit(X, y)
    assert longer.loss_history_[-1] < short.loss_history_[-1]


def test_convergence_to_known_nonseparable_solution():
    # With zero features, the optimum is the observed positive fraction, 1/4.
    model = LogisticRegression(learning_rate=0.5, max_iter=500).fit([0, 0, 0, 0], [0, 0, 0, 1])
    assert_allclose(model.coef_, [0.0], atol=1e-10)
    assert model.intercept_ == pytest.approx(-np.log(3.0), abs=1e-8)
    assert_allclose(model.predict_proba([0]), [[0.75, 0.25]], atol=1e-8)
    expected_loss = -0.75 * np.log(0.75) - 0.25 * np.log(0.25)
    assert model.loss_history_[-1] == pytest.approx(expected_loss, abs=1e-10)


@pytest.mark.parametrize(
    "X, y, message",
    [
        (1.0, [0], "X must be a 1D"),
        (np.ones((2, 1, 1)), [0, 1], "X must be a 1D"),
        ([1, 2], 0.0, "y must be a 1D"),
        ([1, 2], [[0], [1]], "y must be a 1D"),
        ([1, 2], np.ones((2, 1, 1)), "y must be a 1D"),
        ([1, 2], [0], "same number of samples"),
        ([1, 2], [], "same number of samples"),
        ([], [], "at least one sample and one feature"),
        (np.empty((0, 2)), [], "at least one sample and one feature"),
        (np.empty((2, 0)), [0, 1], "at least one sample and one feature"),
    ],
)
def test_fit_rejects_invalid_dimensions(X, y, message):
    with pytest.raises(ValueError, match=message):
        LogisticRegression().fit(X, y)


@pytest.mark.parametrize("y", [[-1, 1], [0, 2], [0, 0.5]])
def test_fit_rejects_nonbinary_labels(y):
    with pytest.raises(ValueError, match="only binary labels 0 and 1"):
        LogisticRegression().fit([1, 2], y)


@pytest.mark.parametrize("y", [[0, 0], [1, 1]])
def test_fit_requires_both_classes(y):
    with pytest.raises(ValueError, match="both binary classes 0 and 1"):
        LogisticRegression().fit([1, 2], y)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["X", "y"])
def test_fit_rejects_nonfinite_values(value, field):
    X, y = [0.0, 1.0], [0.0, 1.0]
    if field == "X":
        X[0] = value
    else:
        y[0] = value
    with pytest.raises(ValueError, match=f"{field} must contain only finite values"):
        LogisticRegression().fit(X, y)


@pytest.mark.parametrize(
    "X, y, field",
    [
        (["bad", "data"], [0, 1], "X"),
        ([0, 1], ["bad", "data"], "y"),
        ([1 + 2j, 2], [0, 1], "X"),
        ([0, 1], [0, 1 + 2j], "y"),
        ([[1], [2, 3]], [0, 1], "X"),
    ],
)
def test_fit_rejects_nonnumeric_complex_or_ragged_input(X, y, field):
    with pytest.raises(ValueError, match=f"{field} must contain real numeric values"):
        LogisticRegression().fit(X, y)


@pytest.mark.parametrize("learning_rate", [0, -0.1, np.nan, np.inf, -np.inf, True, np.bool_(True), "0.1", None, 1j])
def test_invalid_learning_rates(learning_rate):
    with pytest.raises(ValueError, match="learning_rate must be a positive finite number"):
        LogisticRegression(learning_rate=learning_rate)


@pytest.mark.parametrize("max_iter", [0, -1, 1.5, "100", None, True, np.bool_(True), np.nan, np.inf])
def test_invalid_iteration_settings(max_iter):
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        LogisticRegression(max_iter=max_iter)


@pytest.mark.parametrize("fit_intercept", [None, "False", 1])
def test_intercept_option_must_be_boolean(fit_intercept):
    with pytest.raises(ValueError, match="fit_intercept must be a boolean"):
        LogisticRegression(fit_intercept=fit_intercept)


def test_numpy_scalar_settings_are_supported():
    model = LogisticRegression(
        learning_rate=np.float64(0.1), max_iter=np.int64(10), fit_intercept=np.bool_(True)
    ).fit([-1, 1], [0, 1])
    assert_allclose(model.predict([-1, 1]), [0, 1], atol=1e-10)


@pytest.mark.parametrize("method", ["predict", "predict_proba", "score"])
def test_methods_require_fit(method):
    model = LogisticRegression()
    arguments = ([1, 2], [0, 1]) if method == "score" else ([1, 2],)
    with pytest.raises(RuntimeError, match="Call fit"):
        getattr(model, method)(*arguments)


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
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
def test_prediction_methods_reject_invalid_input(fitted_model, method, X, message):
    with pytest.raises(ValueError, match=message):
        getattr(fitted_model, method)(X)


@pytest.mark.parametrize(
    "X, y, message",
    [
        ([0, 1], [[0], [1]], "y must be a 1D"),
        ([0, 1], [1], "same number of samples"),
        ([0, 1], [0, 2], "only binary labels 0 and 1"),
        ([0, 1], [0, np.nan], "y must contain only finite values"),
        ([0, 1], [0, np.inf], "y must contain only finite values"),
        ([0, 1], ["bad", "data"], "y must contain real numeric values"),
        ([[0, 1], [1, 2]], [0, 1], "X must have 1 features"),
    ],
)
def test_score_rejects_invalid_input(fitted_model, X, y, message):
    with pytest.raises(ValueError, match=message):
        fitted_model.score(X, y)


def test_extreme_values_have_stable_probabilities_and_loss():
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        model = LogisticRegression(learning_rate=0.01, max_iter=50).fit(
            [-1000, -500, 500, 1000], [0, 0, 1, 1]
        )
        probabilities = model.predict_proba([-1e300, 0, 1e300])
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.loss_history_))
    assert_allclose(probabilities, [[1, 0], [0.5, 0.5], [0, 1]], atol=1e-12)


def test_training_overflow_has_meaningful_error():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="Scale X or reduce learning_rate"):
        model.fit([1e308, -1e308], [0, 1])
    with pytest.raises(RuntimeError, match="Call fit"):
        model.predict([1])


def test_prediction_overflow_has_meaningful_error(fitted_model):
    with pytest.raises(ValueError, match="Scale X before predicting"):
        fitted_model.predict_proba([np.finfo(float).max])


def test_refitting_starts_from_zero_and_resets_loss_history(fitted_model):
    original_coef = fitted_model.coef_.copy()
    original_losses = fitted_model.loss_history_.copy()
    fitted_model.fit([-3, -2, -1, 1, 2, 3], [0, 0, 0, 1, 1, 1])
    assert_allclose(fitted_model.coef_, original_coef, atol=1e-12)
    assert_allclose(fitted_model.loss_history_, original_losses, atol=1e-12)


def test_refit_updates_feature_count(fitted_model):
    fitted_model.fit([[-1, 0], [0, -1], [1, 0], [0, 1]], [0, 0, 1, 1])
    assert fitted_model.n_features_in_ == 2
    assert_allclose(fitted_model.predict([[-2, 0], [0, 2]]), [0, 1], atol=1e-10)


def test_fit_and_predict_do_not_modify_inputs():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    original_X, original_y = X.copy(), y.copy()
    model = LogisticRegression(max_iter=100).fit(X, y)
    model.predict_proba(X)
    assert_allclose(X, original_X, atol=1e-12)
    assert_allclose(y, original_y, atol=1e-12)
