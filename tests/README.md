# Unit tests

`test_linear_regression.py` checks the implemented NumPy model against
analytical solutions. Coverage includes predictions, multiple features,
optional intercepts, noisy least squares, R², invalid inputs, unfitted models,
rank-deficient and underdetermined designs, and refitting.

Name test files `test_*.py` and import the installed `xinml` package. From the
project root, run `uv run pytest -v`.

Numerical assertions use floating-point tolerances. Tests import the installed
package and do not require scikit-learn. Add real tests alongside future
algorithms; do not add placeholder passing tests.
