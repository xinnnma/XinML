# Notebooks

These notebooks use the installed XinML package and scikit-learn's bundled
datasets. The model implementations are imported from `xinml`; they are not
copied into notebooks or replaced by scikit-learn estimators.

| Notebook | Purpose |
| --- | --- |
| [01_data_exploration.ipynb](01_data_exploration.ipynb) | Explore raw Diabetes features, inspect statistics and missing values, visualize relationships, and demonstrate training-only scaling. |
| [02_linear_regression.ipynb](02_linear_regression.ipynb) | Train XinML Linear Regression on Diabetes and evaluate test R², MSE, RMSE, and prediction plots. |
| [03_logistic_regression.ipynb](03_logistic_regression.ipynb) | Train XinML Logistic Regression on Breast Cancer and inspect probabilities, accuracy, confusion matrix, and cross-entropy loss. |

## Install the environment

From the XinML repository root, run:

```bash
uv sync --locked --dev
```

The development group includes pandas, matplotlib, scikit-learn, nbconvert,
and the existing IPython kernel and pytest tools. NumPy remains the package's
only runtime dependency. No JupyterLab or browser server is needed for VS Code.
Installing dependencies initially needs package-index access; running the
notebooks afterward does not need internet or dataset downloads.

## Run in VS Code

1. Open the XinML folder and use the Python and Jupyter extensions.
2. Open any of the three notebooks.
3. In **Select Kernel**, choose **Python Environments** and the interpreter
   inside XinML's `.venv` folder.
4. Restart the kernel and choose **Run All**. Save the notebook to retain
   updated outputs and figures.

Each notebook runs independently from top to bottom. Read them in numbered
order for the full workflow. If importing `xinml` fails, rerun the setup
command and confirm that the selected kernel belongs to this project's `.venv`.
Do not add source directories to `sys.path` or copy model code into a notebook.

## Optional terminal execution

To make the environment explicit for Jupyter's command-line tools, register
the project kernel on your machine and then execute all three notebooks:

```bash
uv run python -m ipykernel install --user --name xinml --display-name "Python (XinML)"
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=xinml \
  --ExecutePreprocessor.timeout=180 notebooks/*.ipynb
```

Kernel registration is local to your machine. The execution command updates
the saved outputs in the three notebook files. The same notebooks can also
be run interactively in VS Code without registering a command-line kernel.

## Reproducibility and data handling

- Install the locked environment with `uv sync --locked --dev`.
- Each split uses 20% test data and `random_state=42`; classification also uses
  stratification to preserve class proportions.
- Diabetes uses raw features (`scaled=False`) rather than the loader's
  normalization over all observations.
- Exploration uses training rows. Scalers learn training means and standard
  deviations only, then transform the held-out features without refitting.
- XinML models fit training data only. Test targets are used only for
  evaluation, and model settings are fixed before evaluation.
- Breast Cancer labels are **0 = malignant, 1 = benign**; probability columns
  follow that order. Both datasets are bundled with scikit-learn.
- Keep the saved metrics, concise tables, and figures for GitHub viewing.
  Exact plot appearance may vary by platform; numerical results should agree
  within ordinary floating-point tolerances in the locked environment.

These are demonstrations on one fixed split, without cross-validation or
hyperparameter searches. They do not establish performance on new populations.

## Reading the saved results

The notebooks contain eight figures in total: three for exploration, two for
Linear Regression, and three for Logistic Regression.

| Notebook | Saved evaluation results |
| --- | --- |
| Linear Regression | R² **0.4526**, MSE **2900.1936**, and RMSE **53.8534** on 89 test rows. |
| Logistic Regression | Accuracy **96.49%**, or 110 correct predictions out of 114 test rows. |

R² and accuracy measure different tasks. R² compares regression squared error
with a test-mean baseline; accuracy counts correct binary labels. The Breast
Cancer confusion matrix is `[[41, 1], [3, 69]]`, with true labels as rows and
predicted labels as columns in the order malignant, benign. It includes one
malignant case classified as benign, so the overall accuracy does not describe
every kind of error. The training-loss plot shows optimization progress; it
does not measure performance on unseen data.
