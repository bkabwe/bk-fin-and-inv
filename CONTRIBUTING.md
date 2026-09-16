# Contributing

This is a solo-maintained project. This document exists to reduce
bus-factor risk: it captures the test/lint/CI expectations already
described in the [README](README.md#testing--ci) in one place so any future
contributor (human or agent) has a clear checklist before opening a PR.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-api.txt   # only if touching api/
```

See [README.md#installation](README.md#installation) for why there are
three requirements files (`requirements.txt`, `requirements-workflows.txt`,
`requirements-api.txt`) and when each applies.

## Before opening a PR

1. **Run the full test suite** and make sure it passes:
   ```bash
   python -m unittest discover -s tests -p "test_*.py" -v
   ```
   Add or update tests for any behavior you change. New modules under
   `modules/` should get a matching `tests/test_<module>.py`; mock all
   network/external API calls (Polygon, SEC EDGAR, FRED, GitHub Releases,
   Brevo, etc.) — tests must not require network access or secrets.

2. **Lint the Python you touched**:
   ```bash
   pip install ruff
   ruff check .
   ```
   CI enforces `ruff check --select E9,F,B,SIM .` (syntax errors, pyflakes,
   bugbear, simplify) as a hard gate; broader rule sets are being phased in
   incrementally, so also skim `ruff check .` output for anything relevant
   to your change even if the CI gate doesn't cover it yet.

3. **If you changed `frontend/`**, also run:
   ```bash
   cd frontend
   npx tsc --noEmit
   npm test
   npm run build
   ```

4. **If you changed `api/`**, make sure `requirements-api.txt` still installs
   cleanly and the app imports without the Streamlit UI stack (the FastAPI
   backend and Celery workers are not supposed to require `streamlit`).

5. **Pin new dependencies.** `requirements.txt`, `requirements-workflows.txt`,
   and `requirements-api.txt` use exact `==` pins. If you add a new
   dependency, pin it to the version you tested against and verify the full
   test suite still passes with that pin installed in a clean virtualenv.

6. **Scan for secrets** before committing if you touched anything that could
   plausibly contain credentials (env files, workflow YAML, API client code).

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request (`test` + `lint` jobs — see
[README.md#testing--ci](README.md#testing--ci) for details). PRs are
expected to pass CI before merge.

## Code review

This repository uses a `CODEOWNERS` file; all changes require review from
the code owner. Keep PRs focused and incremental — prefer several small PRs
over one large one, especially for changes that touch modeling logic
(`modules/scoring_engine.py`, `modules/fundamental_analysis.py`,
`modules/backtester.py`) since those are the highest-risk areas to regress
silently.
