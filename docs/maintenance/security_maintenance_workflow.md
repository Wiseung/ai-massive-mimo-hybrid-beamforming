# Optional Security Maintenance Workflow

## Current Policy

- security workflow is optional / manual-first
- it does not enter required CI as a blocker
- it does not change the `v1.0.x` research scope

## Recommended Maintenance Items

- CodeQL code scanning
- Dependabot alerts
- dependency review action
- `pip-audit` or `pip check` as a local/manual check

## Why Not Required Yet

- keep CI fast and stable
- Sionna remains an optional dependency
- GPU / Sionna-heavy paths should not block ordinary PRs

## Layering Recommendation

- required CI:
  - `compileall`
  - `pytest`
- optional/manual security checks:
  - CodeQL
  - dependency review
  - `pip-audit`
- local release checklist:
  - release health dashboard
  - optional manual security scan

## Current Non-blocking Warning Interpretation

- if `pip check` passes but `pip-audit` is unavailable:
  - report a warning
  - do not report a blocker
  - recommend `run_manual_audit`

## Manual Follow-up

- existing `pip-audit` path:
  - `python scripts/run_manual_pip_audit.py --out outputs/maintenance/manual_pip_audit.json`
- temporary venv path:
  - `python scripts/run_manual_pip_audit.py --out outputs/maintenance/manual_pip_audit.json --allow-install-in-venv`
- do not pollute the project main environment

## Boundary Statement

- no Sionna RT
- no ray tracing
- no 5G NR full stack
- no full native-only benchmark
- no production e2e claim
