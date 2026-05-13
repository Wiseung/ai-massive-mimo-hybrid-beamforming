# Security And Dependency Maintenance

## Current Dependency Boundary

- Sionna remains an optional dependency.
- Required CI should not force every environment to install Sionna.

## Suggested Manual / Optional Health Checks

- release body consistency audit
- artifact reproducibility audit
- optional Sionna regression monitor
- release tag health audit

## Future Optional Security Maintenance

- dependency review / dependency audit
- CodeQL / code scanning
- Dependabot alerts / security updates

## Scope Note

These are maintenance recommendations only. They do not expand the `v1.0.x` scope.
