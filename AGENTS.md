# Agent Instructions

## Project Status

XinML is a Python 3.11+ course project using a `src/xinml/` package layout, NumPy, uv, Hatchling, and pytest. It includes two algorithms, 152 tests, and three executed notebooks using bundled Diabetes and Breast Cancer datasets. The Phase 5 audit, final documentation, and presentation notes are complete, reviewed, and approved for publication. Preserve both algorithms, their tests, and useful notebook outputs. Subsequent work, commits, and pushes require explicit instructions. Presentation notes and the audit belong in `docs/`; actual presentation delivery remains the student's responsibility.

## Working Rules

- Inspect relevant files and nearby tests before making changes.
- Keep changes focused on the requested task and preserve unrelated user changes.
- Prefer existing project patterns and dependencies over introducing new abstractions or tools.
- Update or add tests when behavior changes, and run the narrowest relevant checks after editing.
- If a check cannot be run, state why and identify the remaining validation needed.
- Do not commit changes or create branches unless explicitly requested.
- Do not add secrets, generated artifacts, or unrelated configuration to the repository.
- Work only in XinML; do not modify the professor's course repository.
- Do not configure a Git email, make a first commit, or publish to GitHub without the user's instructions. The user will provide a Rice email before the first commit.
- Keep real tests in `tests/` and notebooks in `notebooks/`. Do not add placeholder tests that imply algorithms work.
- Notebook scalers and models must fit training rows only. Use installed XinML models; scikit-learn is for bundled datasets and utilities. NumPy remains the only runtime dependency; notebook tools belong in the development group.

## Verified Commands

- `uv sync --locked --dev` installs the package and development dependencies into `.venv`.
- `uv build` builds the source distribution and wheel.
- `uv run pytest -v` runs both models' tests against analytical solutions, deterministic datasets, and invalid inputs. Logistic Regression also checks numerical stability and convergence.
- Use the project `.venv` kernel to execute notebooks from top to bottom. See `notebooks/README.md` for VS Code and command-line setup. Preserve useful outputs and figures, and remove saved errors and unnecessary execution metadata.

## Keeping These Instructions Current

When the project adopts a language, framework, or standard validation commands, update this file with verified project-specific guidance. Keep instructions concise and avoid duplicating details already documented in the project.
