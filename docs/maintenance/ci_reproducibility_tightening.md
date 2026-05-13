# CI Reproducibility Tightening

## Current CI

- `compileall`
- `pytest`

## Not Recommended For Required CI

- long training
- DeepMIMO download
- full Sionna experiment matrix
- GPU-heavy benchmark

## Suggested Manual / Optional Smoke

- stable minimal reproduction
- release body consistency audit
- artifact reproducibility audit
- optional Sionna regression monitor

## CI Layering

- required CI:
  - `compileall`
  - `pytest`
- optional/manual CI:
  - stable minimal reproduction
  - maintenance audits
- local release checklist:
  - release body consistency
  - artifact reproducibility
  - optional Sionna regression monitor

## Dependency Boundary

Keep Sionna optional. CI should not require every environment to install Sionna.
