# Sionna CSI Consumer Artifact Manifest

- generated_from_commit: `87bca20ef1c6712cb8d942804d4af21e96a71196`
- note: consumer unification keeps ExtractedCSI preferred, raw H_f as fallback, and cross-run comparison semantics conservative.

| name | exists | csi_interface_used | same_csi_object_used_for_all_methods | all_consumers_accept_csi | no_new_fallback_introduced | comparison_type | strict_equivalence_claim_allowed | project_h_f_assisted | extracted_h_f_used | command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| csi_consumer_audit | True | True | None | True | True | consumer_audit | False | False | True | `python scripts/audit_csi_consumers.py --out outputs/sionna_channel_extraction/csi_consumer_audit.json` |
| unified_csi_consumers_summary | True | True | True | True | True | single_run_unified_demo | False | False | True | `python scripts/demo_unified_csi_consumers.py --out outputs/sionna_channel_extraction/unified_csi_consumers_summary.json` |
| unified_csi_consumers_metrics | True | True | True | True | True | single_run_unified_demo_metrics | False | False | True | `python scripts/demo_unified_csi_consumers.py --out outputs/sionna_channel_extraction/unified_csi_consumers_summary.json` |
| unified_csi_consumer_comparison | True | True | False | True | True | cross_run_comparison | False | False | True | `python scripts/compare_unified_csi_consumers.py --baseline outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --unified outputs/sionna_channel_extraction/unified_csi_consumers_metrics.csv --out outputs/sionna_channel_extraction` |
| csi_backed_beamforming_summary | True | True | False | True | True | single_run_baseline | False | False | True | `python scripts/sionna_csi_backed_beamforming_chain.py --out outputs/sionna_channel_extraction/csi_backed_beamforming_summary.json --receiver-mode auto --seed 0` |
| csi_interface_audit | True | True | False | None | True | v0_6_csi_audit | False | False | True | `python scripts/audit_sionna_csi_interface.py --out outputs/sionna_channel_extraction/csi_interface_audit.json` |
| csi_same_batch_equivalence | True | True | None | None | True | same_batch_equivalence | True | False | True | `python scripts/validate_csi_same_batch_equivalence.py --out outputs/sionna_channel_extraction/csi_same_batch_equivalence.json` |
