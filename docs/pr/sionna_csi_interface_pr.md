# PR Title Suggestion

`refactor: add provenance-aware CSI interface for Sionna channel extraction`

## Summary

This PR hardens the optional Sionna-native channel-extraction workflow with a provenance-aware CSI interface. The main goal is to replace ad hoc extracted-channel handoff with a standardized `ExtractedCSI` object, then validate that the CSI-backed path is numerically consistent with the earlier raw extracted-H path under a shared realization.

## What Is Included

- `ExtractedCSI` schema and provenance metadata
- CSI provenance audit
- CSI-backed beamforming chain
- same-batch equivalence validation
- raw-vs-CSI mismatch root-cause audit
- corrected cross-run comparison semantics
- CSI artifact manifest
- minimal reproduction command

## What Is Explicitly Not Included

- Sionna RT
- ray tracing
- 5G NR full stack
- full native-only benchmark
- production e2e
- stable learned `> WMMSE-iter5` claim

## Key Results

- CSI audit passed with complete axes and provenance metadata.
- `project_rzf` and `learned_residual_rzf` both consume the CSI object.
- CSI-backed beamforming succeeds on the native receiver path.
- same-batch equivalence passes with zero observed diff in the current checked metrics.

## Same-batch Equivalence Result

- `same_channel_tensor_used = true`
- `same_bits_used = true`
- `same_noise_config_used = true`
- `same_receiver_config_used = true`
- `numeric_consistency_within_tolerance = true`
- `ranking_consistent = true`
- `max_abs_diff_sum_rate = 0.0`
- `max_abs_diff_symbol_mse = 0.0`
- `max_abs_diff_sinr_db = 0.0`

## Previous Mismatch Root Cause

- previous mismatch is now audited as `cross_run_comparison_without_shared_realization`
- `csi_interface_bug_evidence = false`
- the comparison script now marks that path as `comparison_type=cross_run_comparison`
- `not_strict_equivalence_test = true`

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- `python scripts/generate_sionna_csi_interface_artifact_manifest.py --out outputs/sionna_channel_extraction/csi_interface_artifact_manifest.json`
- `python scripts/reproduce_sionna_csi_interface_minimal.py --out outputs/repro/sionna_csi_interface_minimal_summary.json`

## Reproduction Commands

```bash
python scripts/audit_sionna_csi_interface.py \
  --out outputs/sionna_channel_extraction/csi_interface_audit.json

python scripts/sionna_csi_backed_beamforming_chain.py \
  --out outputs/sionna_channel_extraction/csi_backed_beamforming_summary.json \
  --receiver-mode auto \
  --seed 0

python scripts/validate_csi_same_batch_equivalence.py \
  --out outputs/sionna_channel_extraction/csi_same_batch_equivalence.json

python scripts/audit_csi_raw_comparison_mismatch.py \
  --raw outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv \
  --csi outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --out outputs/sionna_channel_extraction

python scripts/generate_sionna_csi_interface_artifact_manifest.py \
  --out outputs/sionna_channel_extraction/csi_interface_artifact_manifest.json

python scripts/reproduce_sionna_csi_interface_minimal.py \
  --out outputs/repro/sionna_csi_interface_minimal_summary.json
```

## Risk Assessment

- low feature risk: this phase is interface/provenance hardening, not a new model rollout
- moderate integration risk: optional Sionna shape assumptions still matter
- low scope risk: RT / ray tracing / 5G NR full stack remain explicitly excluded
- moderate interpretation risk if cross-run comparison is misread as a strict equivalence test

## Merge Recommendation

Merge if the project wants a cleaner optional CSI interface with deterministic shared-realization validation and reviewer-friendly artifacts. Keep the wording conservative: same-batch equivalence passes, but the branch is still not a full native-only benchmark.
