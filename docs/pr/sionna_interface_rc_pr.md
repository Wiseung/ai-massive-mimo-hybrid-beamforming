# PR Title Suggestion

`chore: prepare interface-first Sionna bridge v1.0.0-rc1`

## Summary

This PR prepares `v1.0.0-rc1` as an interface-first Sionna bridge release candidate. The focus is not new models or a full native benchmark claim. The focus is that the interface stack from Sionna channel tensors through `ExtractedCSI`, `PrecoderOutput`, optional native precoder bridging, and the native receiver path is now documented, reproducible, and contract-hardened.

## Release Lineage

- `v0.1.0` benchmark prototype
- `v0.2.0` optional Sionna PHY/OFDM demos
- `v0.3.0` learned beamformer training
- `v0.4.0` native receiver chain
- `v0.5.0` channel extraction bridge
- `v0.6.0` CSI interface
- `v0.7.0` CSI consumer unification
- `v0.8.0` `ExtractedCSI + PrecoderOutput`
- `v0.9.0` optional native precoder bridge
- `v1.0.0-rc1` contract-hardened interface candidate

## What Is Included

- RC artifact manifest across the interface lineage
- RC minimal reproduction
- contract-hardened native precoder schema
- skip/fallback policy hardening
- contract-aware native precoder demo
- contract regression matrix
- README/docs RC cleanup

## What Is Explicitly Not Included

- Sionna RT
- ray tracing
- 5G NR full stack
- full native-only benchmark
- mainline native replacement
- strict `project_rzf` equivalence
- production e2e
- stable learned `> WMMSE-iter5` claim

## Key Results

- `ExtractedCSI` can be created from the Sionna channel tensor path
- `PrecoderOutput` is available across analytic / learned / optional native-method paths
- Sionna `RZFPrecoder` remains callable and convertible
- native receiver path remains successful on the supported interface path
- contract validation passes
- regression matrix passes
- `aliasing_project_rzf_detected = false`

## Contract Validation and Regression Matrix

- `contract_valid = true`
- `relationship_status = close_but_different`
- `strict_equivalence_claim_allowed = false`
- `sionna_native_precoder = true` only for the adapter-generated native object
- failure scenarios keep explicit skip/fallback reasons
- no scenario aliases `project_rzf` as `sionna_rzf_precoder`

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- `python scripts/generate_sionna_interface_rc_artifact_manifest.py --out outputs/sionna_interface_rc/interface_rc_artifact_manifest.json`
- `python scripts/reproduce_sionna_interface_rc_minimal.py --out outputs/repro/sionna_interface_rc_minimal_summary.json`

## Reproduction Commands

```bash
python scripts/generate_sionna_interface_rc_artifact_manifest.py \
  --out outputs/sionna_interface_rc/interface_rc_artifact_manifest.json

python scripts/reproduce_sionna_interface_rc_minimal.py \
  --out outputs/repro/sionna_interface_rc_minimal_summary.json

python scripts/validate_sionna_native_precoder_contract.py \
  --out outputs/sionna_precoder_api/native_precoder_contract_validation.json

python scripts/demo_sionna_native_precoder_contract.py \
  --out outputs/sionna_precoder_api/native_precoder_contract_demo.json

python scripts/test_sionna_native_precoder_contract_matrix.py \
  --out outputs/sionna_precoder_api/native_precoder_contract_matrix.json
```

## Risk Assessment

- low feature risk: this is release-candidate hardening around already validated bridges
- low dependency risk: Sionna remains optional
- moderate interpretation risk if `close_but_different` is overstated as strict equivalence or full native-only

## Merge Recommendation

Merge if the project wants a clean `v1.0.0-rc1` checkpoint for the interface-first Sionna bridge story. Keep the claim narrow: release candidate for interfaces and reproducibility, not production e2e and not a full native-only benchmark.
