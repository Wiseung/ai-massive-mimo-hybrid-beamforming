# PR Title Suggestion

`refactor: add PrecoderOutput bridge for Sionna-native chain`

## Summary

This PR hardens the optional Sionna-native chain by standardizing the output side of the current project-assisted bridge. `ExtractedCSI` remains the preferred input interface, `PrecoderOutput` becomes the preferred output interface, and the native receiver path now consumes the standardized object directly. The goal is interface hardening and reviewer-friendly validation, not a new model claim.

## What Is Included

- `PrecoderOutput` dataclass/schema
- `PrecoderOutput` input/output adapters
- analytic precoder output unification
- learned beamformer output unification
- unified CSI + `PrecoderOutput` demo
- `PrecoderOutput` audit
- same-batch equivalence validation
- previous mismatch root-cause audit
- `PrecoderOutput` artifact manifest
- `PrecoderOutput` minimal reproduction

## What Is Explicitly Not Included

- Sionna RT
- ray tracing
- 5G NR full stack
- full native-only benchmark
- production e2e
- stable learned `> WMMSE-iter5` claim

## Key Results

- `project_rzf` / `project_wmmse_iter_5` emit `PrecoderOutput`
- `learned_residual_rzf` / `learned_residual_wmmse_distill` emit `PrecoderOutput`
- `all_precoders_emit_precoder_output = true`
- `all_receiver_consumers_accept_precoder_output = true`
- `native_receiver_success = true`
- same-batch equivalence passes with zero observed diff in the current checked metrics
- no new fallback is introduced

## Same-batch Equivalence Result

- `same_csi_object_used = true`
- `same_raw_f_f_used = true`
- `same_bits_used = true`
- `same_noise_config_used = true`
- `same_receiver_config_used = true`
- `precoder_output_f_f_matches_raw = true`
- `numeric_consistency_within_tolerance = true`
- `ranking_consistent = true`
- `strict_equivalence_claim_allowed = true`
- `max_abs_diff_raw_f_f_vs_precoder_output = 0.0`
- `max_abs_diff_sum_rate = 0.0`
- `max_abs_diff_symbol_mse = 0.0`
- `max_abs_diff_sinr_db = 0.0`

## Previous Mismatch Root Cause

- the earlier raw-F_f-vs-PrecoderOutput ranking mismatch was `comparison_type=cross_run_comparison`
- `same_seed_used = true`, but `same_csi_object_used = false`
- `same_raw_f_f_used = false`
- `interface_bug_evidence = false`
- the mismatch is therefore treated as independent-run variance, not `PrecoderOutput` bug evidence

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- `python scripts/generate_sionna_precoder_interface_artifact_manifest.py --out outputs/sionna_channel_extraction/precoder_interface_artifact_manifest.json`
- `python scripts/reproduce_sionna_precoder_interface_minimal.py --out outputs/repro/sionna_precoder_interface_minimal_summary.json`

## Reproduction Commands

```bash
python scripts/audit_precoder_interface_consumers.py \
  --out outputs/sionna_channel_extraction/precoder_interface_audit.json

python scripts/demo_unified_csi_and_precoder_interfaces.py \
  --out outputs/sionna_channel_extraction/unified_csi_precoder_summary.json

python scripts/validate_precoder_output_same_batch_equivalence.py \
  --out outputs/sionna_channel_extraction/precoder_output_same_batch_equivalence.json

python scripts/audit_precoder_output_comparison_mismatch.py \
  --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv \
  --out outputs/sionna_channel_extraction

python scripts/generate_sionna_precoder_interface_artifact_manifest.py \
  --out outputs/sionna_channel_extraction/precoder_interface_artifact_manifest.json

python scripts/reproduce_sionna_precoder_interface_minimal.py \
  --out outputs/repro/sionna_precoder_interface_minimal_summary.json
```

## Risk Assessment

- low feature risk: this phase is interface hardening and release preparation
- low regression risk: raw `H_f` / raw `F_f` fallbacks remain for backward compatibility
- low scope risk: RT / ray tracing / 5G NR full stack remain explicitly excluded
- moderate interpretation risk if the cross-run comparison artifact is misread as strict equivalence

## Merge Recommendation

Merge if the project wants a cleaner optional `ExtractedCSI -> PrecoderOutput -> native receiver` bridge with deterministic same-batch validation and reviewer-oriented artifacts. Keep the wording conservative: the same-batch equivalence passes, but the branch is still not a full native-only benchmark.
