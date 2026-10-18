# XinML — Machine Learning from Scratch

XinML is a final project for **CMOR 438 / INDE 577: Data Science & Machine
Learning** at Rice University, taught by Professor Randy R. Davila. The goal is
to build a small, understandable Python machine learning package and explain
its use through Jupyter notebooks.

**Current status: Phase 2 — Linear Regression.** The package provides a NumPy
ordinary least-squares model with pytest tests. Notebooks and a second
algorithm are still planned.

## Planned features

- At least two reusable machine learning algorithms implemented with NumPy.
  Linear Regression is implemented. Logistic Regression remains a provisional
  second choice, pending review against the instructor's course material.
- Notebooks covering data exploration, preprocessing, modeling, and evaluation,
  including at least one practical demonstration of the `xinml` package.
- Comprehensive pytest tests for implemented behavior, numerical results,
  input validation, and relevant edge cases.
- Documentation explaining the algorithms, package installation, and usage.
- A separate public GitHub repository with a clear commit history, followed by
  a final presentation explaining the repository and results.

## Project layout

```text
XinML/
├── AGENTS.md
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── xinml/
│       ├── __init__.py
│       └── linear_regression.py
├── notebooks/
│   └── README.md
└── tests/
    ├── README.md
    └── test_linear_regression.py
```

`src/xinml/` holds reusable package code. `notebooks/` will contain the course
demonstrations, and `tests/` will contain the unit tests. `uv.lock` records
resolved dependency versions for a reproducible environment.

## Local setup

Requires Python **3.11 or newer** and [uv](https://docs.astral.sh/uv/).
Run the following from the XinML project folder:

```bash
uv sync --locked --dev
uv run --locked python -c "import xinml; print(xinml.__file__)"
```

`uv sync` creates a local `.venv` and installs XinML in editable mode along
with NumPy, pytest, and the IPython kernel. Changes to package source files
are then available without reinstalling the package. See the
[uv project documentation](https://docs.astral.sh/uv/concepts/projects/config/)
for packaging details.

In VS Code, use the Python and Jupyter extensions and select the interpreter
or notebook kernel from this project's `.venv`. Notebook tools and test tools
are development dependencies; NumPy is the package's only runtime dependency.

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
models, and rank-deficient designs. Numerical checks use floating-point
tolerances. No scikit-learn dependency is required.

## Next steps

Review the Linear Regression implementation before approving Phase 3.
Notebooks, a second algorithm, GitHub publishing, and the final presentation
remain future work. The initial project setup has a local Git commit; the
Linear Regression changes are left uncommitted for review.
