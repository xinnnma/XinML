# Agent Instructions

## Project Status

XinML is a Python 3.11+ course project using a `src/xinml/` package layout, NumPy, uv, Hatchling, and pytest. Phase 3 adds binary Logistic Regression with batch gradient descent alongside the existing least-squares Linear Regression. Preserve the Linear Regression implementation and tests. Keep Phase 3 changes uncommitted and unpushed for user review; do not create notebooks or proceed to another phase without approval.

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

## Verified Commands

- `uv sync --locked --dev` installs the package and development dependencies into `.venv`.
- `uv build` builds the source distribution and wheel.
- `uv run pytest -v` runs both models' tests against analytical solutions, deterministic datasets, and invalid inputs. Logistic Regression also checks numerical stability and convergence.

## Keeping These Instructions Current

When the project adopts a language, framework, or standard validation commands, update this file with verified project-specific guidance. Keep instructions concise and avoid duplicating details already documented in the project.
