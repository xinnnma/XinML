# XinML — Machine Learning from Scratch

XinML is a final project for **CMOR 438 / INDE 577: Data Science & Machine
Learning** at Rice University, taught by Professor Randy R. Davila. The goal is
to build a small, understandable Python machine learning package and explain
its use through Jupyter notebooks.

The package provides NumPy Linear Regression and binary Logistic Regression.
Three executed notebooks demonstrate exploration, preprocessing, modeling,
and evaluation on held-out data, with saved results and figures.

## Project contents

- Two reusable estimators implemented with NumPy, without scikit-learn models.
- Three independent Jupyter notebooks with eight saved figures.
- 152 pytest tests covering numerical results, input validation, and edge cases.
- Installation and usage examples, a final audit, and presentation notes.

Public repository: [xinnnma/XinML](https://github.com/xinnnma/XinML).

## Project layout

```text
XinML/
├── AGENTS.md
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── FINAL_AUDIT.md
│   └── PRESENTATION.md
├── src/
│   └── xinml/
│       ├── __init__.py
│       ├── linear_regression.py
│       └── logistic_regression.py
├── notebooks/
│   ├── README.md
│   ├── 01_data_exploration.ipynb
│   ├── 02_linear_regression.ipynb
│   └── 03_logistic_regression.ipynb
└── tests/
    ├── README.md
    ├── test_linear_regression.py
    └── test_logistic_regression.py
```

`src/xinml/` holds reusable package code. `notebooks/` contains the course
demonstrations, and `tests/` contains the unit tests. `uv.lock` records
resolved dependency versions for a reproducible environment.

## Local setup

Requires Python **3.11 or newer** and [uv](https://docs.astral.sh/uv/).
For a new checkout:

```bash
git clone https://github.com/xinnnma/XinML.git
cd XinML
```

From the XinML project folder, install the locked environment:

```bash
uv sync --locked --dev
uv run --locked python -c "import xinml; print(xinml.__file__)"
```

`uv sync` creates a local `.venv` and installs XinML in editable mode along
with NumPy and the development tools. Changes to package source files
are then available without reinstalling the package. See the
[uv project documentation](https://docs.astral.sh/uv/concepts/projects/config/)
for packaging details.

In VS Code, use the Python and Jupyter extensions and select the interpreter
or notebook kernel from this project's `.venv`. Notebook tools and test tools
are development dependencies; NumPy is the package's only runtime dependency.
The development group also includes pandas, matplotlib, scikit-learn, and
nbconvert for notebook analysis, plots, bundled datasets, and execution.

## Linear Regression

`LinearRegression` fits the model `y = X @ coef_ + intercept_` by minimizing
the sum of squared prediction errors. With `fit_intercept=True` (the default),
the design matrix adds a column of ones for the intercept. With
`fit_intercept=False`, the intercept is fixed at zero.

Training uses [`numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
without scikit-learn. This avoids explicitly inverting a matrix and handles
rank-deficient or underdetermined designs by selecting the minimum-norm
least-squares solution. With redundant features, coefficients need not be
unique; interpret them with care.

After local setup, run this example in Python:

```python
import numpy as np
from xinml import LinearRegression

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2x + 1

model = LinearRegression().fit(X, y)
predictions = model.predict([[4.0], [5.0]])
r_squared = model.score(X, y)

np.testing.assert_allclose(model.coef_, [2.0], atol=1e-10)
np.testing.assert_allclose(model.intercept_, 1.0, atol=1e-10)
np.testing.assert_allclose(predictions, [9.0, 11.0], atol=1e-10)
np.testing.assert_allclose(r_squared, 1.0, atol=1e-10)
print(predictions)  # [ 9. 11.]
print(r_squared)    # 1.0
```

- `fit(X, y)` trains the model and returns `self`. Learned attributes are
  `coef_`, `intercept_`, and `n_features_in_`.
- `predict(X)` returns a one-dimensional array of predictions.
- `score(X, y)` computes R² as `1 - residual_sum_of_squares / total_sum_of_squares`.
  A perfect fit scores 1.0; scores can be negative. R² is undefined for fewer
  than two samples or a constant target; this implementation raises
  `ValueError` in those cases.

Use finite, real numeric values. `X` may have shape `(n_samples, n_features)`
or `(n_samples,)` for a single feature. For several features in one sample,
use a 2D row such as `[[x1, x2]]`. `y` must have shape `(n_samples,)`, with
one target per sample. Empty inputs, incompatible shapes, nonnumeric values,
complex values, NaN, and infinity raise `ValueError`. Prediction requires the
same number of features as training. Calling `predict` or `score` before
`fit` raises `RuntimeError`.

## Logistic Regression

`LogisticRegression` performs binary classification with labels 0 and 1. A
linear score `z = X @ coef_ + intercept_` is mapped to the probability of
class 1 by the sigmoid function `p = 1 / (1 + exp(-z))`. Predictions use
class 1 when `p >= 0.5`, including ties, and class 0 otherwise.

Training uses batch gradient descent to minimize mean binary cross-entropy,
`mean(-y*log(p) - (1-y)*log(1-p))`. Starting from zero weights, each update
subtracts the learning rate times the gradient `A.T @ (p-y) / n_samples`.
Here `A` is the feature matrix with an optional column of ones for the
intercept. All samples contribute to every update; no randomness is used.

The sigmoid uses separate branches to avoid exponential overflow. Loss is
calculated from logits with
[`numpy.logaddexp`](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
to avoid taking the logarithm of a probability rounded to zero.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `learning_rate` | `0.1` | Positive, finite gradient descent step size. |
| `max_iter` | `1000` | Positive integer number of updates. |
| `fit_intercept` | `True` | Learn an intercept; `False` fixes it at zero. |

```python
import numpy as np
from xinml import LogisticRegression

X = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression(learning_rate=0.1, max_iter=1000).fit(X, y)
probabilities = model.predict_proba([[-2.0], [2.0]])
predictions = model.predict([[-2.0], [2.0]])
accuracy = model.score(X, y)

assert probabilities.shape == (2, 2)  # Columns are P(class 0), P(class 1).
np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0], atol=1e-12)
np.testing.assert_allclose(predictions, [0, 1], atol=1e-12)
np.testing.assert_allclose(accuracy, 1.0, atol=1e-12)
assert model.loss_history_[-1] < model.loss_history_[0]
print(predictions)  # [0 1]
print(accuracy)     # 1.0
```

- `fit(X, y)` returns `self` and learns `coef_`, `intercept_`, and
  `n_features_in_`. `loss_history_` contains the initial loss followed by the
  loss after each update, for a total of `max_iter + 1` values.
- `predict_proba(X)` returns shape `(n_samples, 2)`, with probability columns
  ordered as class 0 and class 1. Each row sums to one.
- `predict(X)` returns an integer array of shape `(n_samples,)`.
- `score(X, y)` returns classification accuracy: the fraction of correct
  predictions. This differs from Linear Regression's R² score.

Input shapes and finite numeric checks match Linear Regression. Training
requires both classes 0 and 1; evaluation labels may contain only one class.
Invalid labels or parameters raise `ValueError`. Prediction methods and
scoring raise `RuntimeError` before fitting. Both models are available with
`from xinml import LinearRegression, LogisticRegression`.

This model has a linear decision boundary and supports binary targets only.
It does not include regularization, automatic feature scaling, or early
stopping. Scale features when their magnitudes differ substantially and tune
the learning rate and iteration count. A fixed iteration count does not
guarantee convergence; on perfectly separable data, unregularized weights
can keep growing as loss decreases. Arithmetic overflow beyond floating-point
range raises `ValueError` with guidance to scale inputs or reduce the step size.

## Notebook demonstrations

| Notebook | Dataset | Demonstrates |
| --- | --- | --- |
| [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | Diabetes | Dimensions, descriptive statistics, missing values, plots, splitting, and scaling. |
| [02_linear_regression.ipynb](notebooks/02_linear_regression.ipynb) | Diabetes | XinML least squares, R², MSE, RMSE, and actual-versus-predicted values. |
| [03_logistic_regression.ipynb](notebooks/03_logistic_regression.ipynb) | Breast Cancer | XinML binary classification, probabilities, accuracy, confusion matrix, and training loss. |

Install all notebook dependencies with `uv sync --locked --dev`. In VS Code,
open XinML, use the Python and Jupyter extensions, open a notebook, and choose
this project's `.venv` interpreter in **Select Kernel**. Restart the kernel
and select **Run All** to reproduce its outputs. Each notebook is independent;
the numerical order is the suggested reading order.

Both datasets are bundled with scikit-learn. After the environment is
installed, notebook execution requires no internet or external dataset files.
Splits use `random_state=42`. Diabetes is loaded with `scaled=False`, and all
scalers and models are fitted only on training rows. The Breast Cancer split
is stratified. Scikit-learn supplies datasets and utilities; XinML supplies
both trained models. Outputs and figures are saved for GitHub viewing.

See [notebooks/README.md](notebooks/README.md) for terminal execution,
environment setup, and reproducibility details. These demonstrations use one
fixed train/test split, so their scores do not measure performance across all
possible splits or populations.

### Recorded results

| Model and dataset | Training / test rows | Held-out results |
| --- | --- | --- |
| Linear Regression — Diabetes | 353 / 89 | R² **0.4526**, MSE **2900.1936**, RMSE **53.8534**. |
| Logistic Regression — Breast Cancer | 455 / 114 | Accuracy **96.49%** (110 of 114 predictions correct). |

R² compares squared prediction error with predicting the test target mean;
accuracy measures the fraction of correct class labels. These scores describe
different tasks and should not be compared directly. RMSE is expressed in the
Diabetes target's units, while MSE uses squared units. The classification
confusion matrix is `[[41, 1], [3, 69]]`, with true classes as rows and
predicted classes as columns, ordered malignant then benign. The one malignant
case predicted as benign is an important error despite the high accuracy.
The classification example is educational and does not validate clinical use.

## Validation and tests

To build the package locally:

```bash
uv build
```

Run the tests:

```bash
uv run pytest -v
```

Tests cover known coefficients, unseen predictions, multiple features,
intercept handling, noisy least squares, R², input validation, unfitted
models, and rank-deficient designs. Logistic Regression tests cover binary
training, probability shapes and class ordering, accuracy, optional
intercepts, input and parameter validation, numerical stability, and
convergence against a known solution. Numerical checks use floating-point
tolerances and deterministic datasets. The models and unit tests do not rely
on scikit-learn; the notebooks use it for datasets and utilities.

## Presentation and audit

[Presentation notes](docs/PRESENTATION.md) provide a 3–5 minute walkthrough
and answers to likely questions. The [final audit](docs/FINAL_AUDIT.md) maps
the supplied requirements to repository evidence and records validation and
remaining submission tasks. Preparing these notes does not replace delivering
the final presentation.
