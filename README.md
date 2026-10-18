# XinML — Machine Learning from Scratch

XinML is a final project for **CMOR 438 / INDE 577: Data Science & Machine
Learning** at Rice University, taught by Professor Randy R. Davila. The goal is
to build a small, understandable Python machine learning package and explain
its use through Jupyter notebooks.

**Current status: Phase 1 — project setup only.** The package scaffold is in
place. No machine learning algorithms, notebooks, or unit tests have been
implemented yet.

## Planned features

- At least two reusable machine learning algorithms implemented with NumPy.
  Linear Regression and Logistic Regression are provisional choices; confirm
  their scope against the instructor's course material before implementation.
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
│       └── __init__.py
├── notebooks/
│   └── README.md
└── tests/
    └── README.md
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

## Validation and future tests

To build the package locally:

```bash
uv build
```

Once real tests have been added, run:

```bash
uv run --locked pytest
```

During Phase 1, pytest reports **no tests collected** and exits with code 5.
That is expected for this scaffold and does not mean any algorithm has been
tested. No placeholder tests are included.

## Next steps

Phase 2 requires approval before work begins. Its first step is to confirm
algorithm choices and expected behavior against the course material, then
implement and test the agreed functionality.

The local Git repository starts without commits or a remote. Provide a Rice
email address before the first commit. Publishing the required public GitHub
repository and preparing the presentation are later steps.
