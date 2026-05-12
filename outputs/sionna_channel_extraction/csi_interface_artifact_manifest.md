# Sionna CSI Interface Artifact Manifest

- generated_from_commit: `2fbc2c5150a88effb95e2b6ee041f17a5f9862a0`
- note: optional Sionna CSI-interface artifacts only; same-batch equivalence and cross-run comparison are tracked separately.

| name | exists | csi_interface_used | same_batch_equivalence | numeric_consistency | ranking_consistent | comparison_type | project_h_f_assisted | extracted_h_f_used | command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| channel_tensor_audit | True | False | None | None | None | channel_tensor_audit | True | False | `python scripts/audit_sionna_channel_tensor_shapes.py --out outputs/sionna_channel_extraction/channel_tensor_audit.json` |
| extract_h_f_demo_summary | True | True | None | None | None | extraction_demo | False | True | `python scripts/sionna_extract_channel_hf_demo.py --out outputs/sionna_channel_extraction/extract_h_f_demo_summary.json` |
| hf_axis_validation | True | True | None | None | None | axis_validation | False | True | `python scripts/validate_sionna_extracted_hf_axes.py --out outputs/sionna_channel_extraction/hf_axis_validation.json` |
| native_channel_beamforming_summary | True | True | None | None | None | native_channel_assisted_summary | False | True | `python scripts/sionna_native_channel_beamforming_chain.py --out outputs/sionna_channel_extraction/native_channel_beamforming_summary.json --receiver-mode auto` |
| csi_interface_audit | True | True | None | None | None | csi_provenance_audit | False | True | `python scripts/audit_sionna_csi_interface.py --out outputs/sionna_channel_extraction/csi_interface_audit.json` |
| csi_backed_beamforming_summary | True | True | None | None | None | single_run_summary | False | True | `python scripts/sionna_csi_backed_beamforming_chain.py --out outputs/sionna_channel_extraction/csi_backed_beamforming_summary.json --receiver-mode auto --seed 0` |
| csi_interface_comparison | True | True | False | False | False | cross_run_comparison | False | True | `python scripts/compare_csi_backed_vs_raw_extracted_h.py --raw outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv --csi outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --out outputs/sionna_channel_extraction` |
| csi_same_batch_equivalence | True | True | True | True | True | same_batch_equivalence | False | True | `python scripts/validate_csi_same_batch_equivalence.py --out outputs/sionna_channel_extraction/csi_same_batch_equivalence.json` |
| csi_raw_mismatch_audit | True | True | False | None | None | cross_run_comparison_audit | False | True | `python scripts/audit_csi_raw_comparison_mismatch.py --raw outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv --csi outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --out outputs/sionna_channel_extraction` |
