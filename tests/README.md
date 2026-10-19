# Unit tests

`test_linear_regression.py` checks the implemented NumPy model against
analytical solutions. Coverage includes predictions, multiple features,
optional intercepts, noisy least squares, R², invalid inputs, unfitted models,
rank-deficient and underdetermined designs, and refitting.

`test_logistic_regression.py` checks binary training, predictions, probability
shapes and class ordering, accuracy, optional intercepts, label and parameter
validation, numerical stability, refitting, and convergence to a known
nonseparable solution. Datasets are deterministic; numerical assertions use
tolerances. The existing Linear Regression tests remain unchanged.

Name test files `test_*.py` and import the installed `xinml` package. From the
project root, run `uv run pytest -v`.

Numerical assertions use floating-point tolerances. Tests import the installed
package and do not require scikit-learn. Add real tests alongside future
algorithms; do not add placeholder passing tests.
