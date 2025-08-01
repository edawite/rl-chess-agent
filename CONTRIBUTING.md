# Contributing Guidelines

Thanks for taking the time to contribute to this reinforcement‑learning chess
agent.  The following guidelines help ensure that the codebase remains
consistent, readable and easy to maintain.

## Code Style

- Follow **PEP 8** conventions throughout the code.  Use descriptive names
  for variables and functions; avoid abbreviations except for well‑known
  terms (e.g. `env`, `elo`).
- Include **type hints** on all function signatures and module‑level
  variables.  This aids both readability and static analysis.
- Write concise **docstrings** for all public functions, classes and
  modules.  Use the Google or NumPy docstring style for clarity.
- Keep functions short and focused; if a function grows beyond ~50 lines
  consider refactoring.
- Avoid generic “helper” functions; name functions according to their
  purpose.
- Add comments sparingly to clarify intent when the code is not
  self‑explanatory, but prefer expressive code over comments.

## Formatter & Linting

This project uses the following tools.  Run them before submitting a pull
request:

- [**black**](https://black.readthedocs.io/) for automatic formatting.  The
  default line length of 88 characters is used.
- [**flake8**](https://flake8.pycqa.org/) for linting; configure it via
  `setup.cfg` if needed.
- [**isort**](https://pycqa.github.io/isort/) for import sorting.

You can install all development dependencies with:

```bash
pip install -r requirements.txt
pip install black flake8 isort
```

Then format and lint your changes:

```bash
isort .
black .
flake8
```

## Testing

Tests live in the `tests/` directory and are written with
[**pytest**](https://docs.pytest.org/).  Ensure that new functionality is
covered by unit tests and that the entire suite passes:

```bash
pytest
```

When adding tests, favour clear assertions over complex logic.  If tests
depend on randomness, seed the random number generator to make them
deterministic.

## Commit Messages

Adopt the **Conventional Commits** style for commit messages.  Use the
summary line to concisely describe what changed and why.  Examples:

- `env: add stalemate detection and material diff reward`
- `train: log per‑episode returns to CSV`
- `docs: expand quickstart section in README`

Avoid vague messages like “fix stuff”.

## Opening Pull Requests

Before opening a pull request:

1. Ensure your branch is up to date with `main` and resolves cleanly.
2. Verify that all tests pass locally and that linting/formatting checks
   succeed.
3. Include a clear description of the changes and the motivation behind
   them.

Following these guidelines helps maintain a high‑quality codebase and makes
it easier for reviewers to provide timely feedback.  Thank you for your
contributions!
