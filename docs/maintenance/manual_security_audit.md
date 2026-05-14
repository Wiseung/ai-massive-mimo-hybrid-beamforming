# Manual Security Audit

## Why The Current Status Can Be Warning

- `pip check` may pass
- `pip-audit` may be unavailable on some machines
- or a manual `pip-audit` run may find vulnerabilities that still require review

This is still a warning path by default, not an automatic blocker.

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
  - keep `warning` unless a true blocker is confirmed
  - use `recommended_next_action = review_dependency_alerts`
- `pip-audit` unavailable:
  - keep `warning`
  - use `recommended_next_action = install_pip_audit_and_rerun`

## Boundary Statement

- optional/manual-first only
- no required CI change
- no scope expansion
- no new release unless a real blocker is found
