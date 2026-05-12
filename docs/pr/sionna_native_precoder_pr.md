# PR Title Suggestion

`probe: add Sionna RZFPrecoder optional native method bridge`

## Summary

This PR hardens the optional Sionna-native precoder bridge after the `v0.8.0` `ExtractedCSI + PrecoderOutput` work. The goal is to document and validate how the currently installed Sionna `RZFPrecoder` can be called, converted into `PrecoderOutput`, and inserted into the existing native receiver path without overstating the integration status.

## What Is Included

- Sionna native precoder API audit
- `RZFPrecoder` adapter bridge
- `sionna_rzf_precoder` optional native method
- same-realization project-vs-Sionna RZF validation
- quick SNR/seed alignment sweep
- updated comparison report
- native precoder artifact manifest
- native precoder minimal reproduction

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

- `sionna_rzf_available = true`
- `sionna_rzf_callable = true`
- `converted_to_precoder_output = true`
- `native_receiver_success = true`
- `sionna_native_precoder = true` for the adapter-generated native output
- `project_side_precoder = false` for the adapter-generated native output
- `relationship_status = close_but_different`
- `strict_equivalence_claim_allowed = false`

## Same-realization Validation Result

- one shared `ExtractedCSI` object is reused
- one shared symbol batch is reused
- one shared native receiver config is reused
- one shared noise config is reused
- semantic compatibility passes
- strict equivalence does not pass
- current observed same-realization diffs:
  - `max_abs_diff_f_f_if_comparable = 0.05982483550906181`
  - `abs_diff_sum_rate = 0.0704193115234375`
  - `abs_diff_symbol_mse = 0.0005603842437267303`
  - `abs_diff_sinr_db = 0.055327415466308594`

## Quick SNR/seed Alignment Result

- seeds: `1,2,3`
- snrs: `0,5,10,15,20 dB`
- callable on all evaluated rows
- converted to `PrecoderOutput` on all evaluated rows
- native receiver success on all evaluated rows
- semantic compatibility on all evaluated rows
- majority relationship remains `close_but_different`
- no run supports a strict equivalence claim

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- `python scripts/generate_sionna_native_precoder_artifact_manifest.py --out outputs/sionna_precoder_api/native_precoder_artifact_manifest.json`
- `python scripts/reproduce_sionna_native_precoder_minimal.py --out outputs/repro/sionna_native_precoder_minimal_summary.json`

## Reproduction Commands

```bash
python scripts/audit_sionna_native_precoder_api.py \
  --out outputs/sionna_precoder_api/native_precoder_api_audit.json

python scripts/probe_sionna_rzf_precoder_bridge.py \
  --out outputs/sionna_precoder_api/rzf_precoder_probe_summary.json

python scripts/validate_sionna_rzf_same_realization.py \
  --out outputs/sionna_precoder_api/sionna_rzf_same_realization.json

python scripts/benchmark_sionna_rzf_precoder_alignment.py \
  --quick \
  --seeds 1 2 3 \
  --snrs 0 5 10 15 20 \
  --out outputs/sionna_precoder_api/sionna_rzf_alignment_quick

python scripts/demo_unified_csi_and_precoder_interfaces.py \
  --out outputs/sionna_channel_extraction/unified_csi_precoder_summary.json \
  --include-sionna-rzf

python scripts/generate_sionna_native_precoder_artifact_manifest.py \
  --out outputs/sionna_precoder_api/native_precoder_artifact_manifest.json

python scripts/reproduce_sionna_native_precoder_minimal.py \
  --out outputs/repro/sionna_native_precoder_minimal_summary.json
```

## Risk Assessment

- low feature risk: this phase is release hardening around an already validated optional bridge
- low dependency risk: Sionna remains optional and all tests keep skip behavior when absent
- moderate interpretation risk if `close_but_different` is misreported as strict equivalence or full native-only

## Merge Recommendation

Merge if the project wants a reviewer-friendly optional native precoder bridge with explicit audit artifacts and conservative wording. Keep the interpretation narrow: callable native API, successful adapter bridge, optional method integration, but not strict equivalence and not full native-only.
