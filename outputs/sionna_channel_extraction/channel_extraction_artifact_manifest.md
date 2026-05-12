# Sionna Channel Extraction Artifact Manifest

- generated_from_commit: `fba7e6b89c0b4c7dc24f49ed246907ab3feb7640`
- note: optional Sionna channel-extraction artifacts only; quick/limited benchmarks are marked explicitly.

| name | exists | extraction_success | native_receiver_success | project_h_f_assisted | extracted_h_f_used | quick_or_full | command |
| --- | --- | --- | --- | --- | --- | --- | --- |
| channel_tensor_audit | True | True | None | None | None | full | `python scripts/audit_sionna_channel_tensor_shapes.py --out outputs/sionna_channel_extraction/channel_tensor_audit.json` |
| extract_h_f_demo_summary | True | True | None | None | True | full | `python scripts/sionna_extract_channel_hf_demo.py --out outputs/sionna_channel_extraction/extract_h_f_demo_summary.json` |
| hf_axis_validation | True | True | None | None | True | full | `python scripts/validate_sionna_extracted_hf_axes.py --out outputs/sionna_channel_extraction/hf_axis_validation.json` |
| native_channel_beamforming_summary | True | True | True | False | True | full | `python scripts/sionna_native_channel_beamforming_chain.py --out outputs/sionna_channel_extraction/native_channel_beamforming_summary.json --receiver-mode auto` |
| extracted_h_consistency_summary | True | True | True | False | True | quick | `python scripts/benchmark_sionna_extracted_h_consistency.py --out outputs/sionna_channel_extraction/extracted_h_consistency --seeds 1 2 3 --snrs 0 5 10 15 20 --quick` |
| extracted_h_consistency_metrics | True | True | True | False | True | quick | `python scripts/benchmark_sionna_extracted_h_consistency.py --out outputs/sionna_channel_extraction/extracted_h_consistency --seeds 1 2 3 --snrs 0 5 10 15 20 --quick` |
| extraction_config_sweep | True | True | None | False | True | quick | `python scripts/sweep_sionna_channel_extraction_config.py --quick --out outputs/sionna_channel_extraction/extraction_config_sweep` |
| project_vs_extracted_hf_comparison | True | True | True | False | True | mixed | `python scripts/compare_project_hf_vs_extracted_hf.py --project outputs/sionna_native_chain/learned_beamforming_receiver_metrics.csv --extracted outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv --consistency outputs/sionna_channel_extraction/extracted_h_consistency/metrics.csv --out outputs/sionna_channel_extraction/project_vs_extracted_hf` |
