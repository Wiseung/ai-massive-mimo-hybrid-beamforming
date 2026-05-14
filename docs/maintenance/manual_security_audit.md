# Manual Security Audit

## Why The Current Status Is Warning

- `pip check` passed
- `pip-audit` is unavailable on the current machine
- vulnerability audit coverage is therefore incomplete

This is a non-blocking warning, not a blocker.

## How To Run The Manual Audit

Existing `pip-audit` path:

```bash
python scripts/run_manual_pip_audit.py \
  --out outputs/maintenance/manual_pip_audit.json
```

Temporary venv path:

```bash
python scripts/run_manual_pip_audit.py \
  --out outputs/maintenance/manual_pip_audit.json \
  --allow-install-in-venv
```

## Environment Guidance

- do not pollute the project main environment
- prefer an isolated temporary venv when manual installation is allowed
- do not turn this into required CI

## Result Interpretation

- no vulnerabilities:
  - dashboard can move to `ok`
- vulnerabilities found:
  - review / fix
- `pip-audit` unavailable:
  - keep `warning`
  - keep `recommended_next_action = install_pip_audit_and_rerun`

## Boundary Statement

- optional/manual-first only
- no required CI change
- no scope expansion
- no new release unless a real blocker is found
