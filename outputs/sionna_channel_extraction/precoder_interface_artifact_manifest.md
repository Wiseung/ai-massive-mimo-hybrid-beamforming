# Sionna Precoder Interface Artifact Manifest

- generated_from_commit: `9ab673ae2168b914251ca7746b775b44da915018`
- note: optional Sionna PrecoderOutput artifacts only; same-batch equivalence and cross-run comparison are tracked separately.

| name | exists | csi_interface_used | precoder_interface_used | same_csi_object_used_for_all_methods | all_precoders_emit_precoder_output | all_receiver_consumers_accept_precoder_output | same_batch_equivalence | numeric_consistency_within_tolerance | strict_equivalence_claim_allowed | comparison_type | command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| precoder_interface_audit | True | True | True | None | True | True | False | None | False | precoder_audit | `python scripts/audit_precoder_interface_consumers.py --out outputs/sionna_channel_extraction/precoder_interface_audit.json` |
| unified_csi_precoder_summary | True | True | True | True | True | True | False | None | False | single_run_unified_demo | `python scripts/demo_unified_csi_and_precoder_interfaces.py --out outputs/sionna_channel_extraction/unified_csi_precoder_summary.json` |
| unified_csi_precoder_metrics | True | True | True | True | True | True | False | None | False | single_run_unified_demo_metrics | `python scripts/demo_unified_csi_and_precoder_interfaces.py --out outputs/sionna_channel_extraction/unified_csi_precoder_summary.json` |
| precoder_output_comparison | True | True | True | False | True | True | False | False | False | cross_run_comparison | `python scripts/compare_raw_ff_vs_precoder_output.py --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv --out outputs/sionna_channel_extraction` |
| precoder_output_same_batch_equivalence | True | True | True | True | True | True | True | True | True | same_batch_equivalence | `python scripts/validate_precoder_output_same_batch_equivalence.py --out outputs/sionna_channel_extraction/precoder_output_same_batch_equivalence.json` |
| precoder_output_mismatch_audit | True | True | True | False | True | True | False | False | False | cross_run_comparison | `python scripts/audit_precoder_output_comparison_mismatch.py --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv --out outputs/sionna_channel_extraction` |
| csi_interface_audit | True | True | False | None | None | None | False | None | False | v0_6_csi_audit | `python scripts/audit_sionna_csi_interface.py --out outputs/sionna_channel_extraction/csi_interface_audit.json` |
| csi_same_batch_equivalence | True | True | False | None | None | None | True | True | True | v0_6_same_batch_equivalence | `python scripts/validate_csi_same_batch_equivalence.py --out outputs/sionna_channel_extraction/csi_same_batch_equivalence.json` |
| csi_consumer_audit | True | True | False | None | None | None | False | None | False | v0_7_csi_consumer_audit | `python scripts/audit_csi_consumers.py --out outputs/sionna_channel_extraction/csi_consumer_audit.json` |
