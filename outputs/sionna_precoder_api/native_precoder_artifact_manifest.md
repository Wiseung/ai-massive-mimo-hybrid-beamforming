# Sionna Native Precoder Artifact Manifest

- generated_from_commit: `30f0fbd6ddda74d72a482064c27c6285e89629fb`
- note: optional native-precoder bridge only; not full native-only and not strict project_rzf equivalence.

| name | exists | sionna_rzf_available | sionna_rzf_callable | converted_to_precoder_output | native_receiver_success | sionna_native_precoder | project_side_precoder | relationship_status | strict_equivalence_claim_allowed | command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_precoder_api_audit | True | True | True | True | True | True | False | close_but_different | False | `python scripts/audit_sionna_native_precoder_api.py --out outputs/sionna_precoder_api/native_precoder_api_audit.json` |
| rzf_precoder_probe_summary | True | True | True | True | True | True | False | close_but_different | False | `python scripts/probe_sionna_rzf_precoder_bridge.py --out outputs/sionna_precoder_api/rzf_precoder_probe_summary.json` |
| sionna_rzf_same_realization | True | True | True | True | True | True | False | close_but_different | False | `python scripts/validate_sionna_rzf_same_realization.py --out outputs/sionna_precoder_api/sionna_rzf_same_realization.json` |
| sionna_rzf_alignment_quick | True | True | True | True | True | True | False | close_but_different | False | `python scripts/benchmark_sionna_rzf_precoder_alignment.py --quick --seeds 1 2 3 --snrs 0 5 10 15 20 --out outputs/sionna_precoder_api/sionna_rzf_alignment_quick` |
| project_vs_sionna_precoder_comparison_v2 | True | True | True | True | True | True | False | close_but_different | False | `python scripts/compare_project_vs_sionna_precoder.py --same-realization outputs/sionna_precoder_api/sionna_rzf_same_realization.json --alignment outputs/sionna_precoder_api/sionna_rzf_alignment_quick/metrics.csv --unified outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv --out outputs/sionna_precoder_api` |
| unified_csi_precoder_summary_with_sionna_rzf | True | True | True | True | True | True | False | close_but_different | False | `python scripts/demo_unified_csi_and_precoder_interfaces.py --out outputs/sionna_channel_extraction/unified_csi_precoder_summary.json --include-sionna-rzf` |
