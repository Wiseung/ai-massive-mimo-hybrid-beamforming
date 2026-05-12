# PR Title Suggestion

`refactor: unify CSI consumers across native Sionna chain`

## Summary

This PR hardens the optional Sionna-native channel-extraction workflow by making `ExtractedCSI` the preferred input interface across analytic precoders, learned beamformers, native-chain scripts, and comparison helpers. The goal is interface unification and provenance clarity, not a new model result claim.

## What Is Included

- CSI consumer audit
- CSI input adapter utilities
- analytic consumer unification
- learned consumer unification
- unified CSI consumer demo
- unified-vs-baseline comparison semantics
- CSI consumer artifact manifest
- CSI consumer minimal reproduction

## What Is Explicitly Not Included

- Sionna RT
- ray tracing
- 5G NR full stack
- full native-only benchmark
- production e2e
- stable learned `> WMMSE-iter5` claim

## Key Results

- `total_consumers_audited = 15`
- `raw_only_high_priority_paths = 0`
- `all_consumers_accept_csi = true`
- `same_csi_object_used_for_all_methods = true`
- `native_receiver_success = true`
- `teacher_used_during_inference = false`
- `no_new_fallback_introduced = true`

## Cross-run Comparison Caveat

- unified-vs-baseline is explicitly `comparison_type=cross_run_comparison`
- `same_seed_used = true`, but `same_csi_tensor_signature = false`
- `strict_equivalence_claim_allowed = false`
- strict same-batch equivalence remains the earlier `v0.6.0` validation path, not this `v0.7.0` cross-run comparison

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- `python scripts/generate_sionna_csi_consumer_artifact_manifest.py --out outputs/sionna_channel_extraction/csi_consumer_artifact_manifest.json`
- `python scripts/reproduce_sionna_csi_consumer_minimal.py --out outputs/repro/sionna_csi_consumer_minimal_summary.json`

## Reproduction Commands

```bash
python scripts/audit_csi_consumers.py \
  --out outputs/sionna_channel_extraction/csi_consumer_audit.json

python scripts/demo_unified_csi_consumers.py \
  --out outputs/sionna_channel_extraction/unified_csi_consumers_summary.json

python scripts/compare_unified_csi_consumers.py \
  --baseline outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --unified outputs/sionna_channel_extraction/unified_csi_consumers_metrics.csv \
  --out outputs/sionna_channel_extraction

python scripts/generate_sionna_csi_consumer_artifact_manifest.py \
  --out outputs/sionna_channel_extraction/csi_consumer_artifact_manifest.json

python scripts/reproduce_sionna_csi_consumer_minimal.py \
  --out outputs/repro/sionna_csi_consumer_minimal_summary.json
```

## Risk Assessment

- low feature risk: this phase is interface unification and documentation hardening
- low regression risk: raw `H_f` fallback remains for backward compatibility
- low scope risk: RT / ray tracing / 5G NR full stack remain explicitly excluded
- moderate interpretation risk if the cross-run comparison is misread as strict equivalence

## Merge Recommendation

Merge if the project wants `ExtractedCSI` to be the preferred interface across current key consumers while preserving raw-`H_f` compatibility. Keep the wording conservative: unified consumer support is real, but the branch is still not a full native-only benchmark, and the unified-vs-baseline artifact is cross-run only.
