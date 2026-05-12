# CSI Consumer Audit

- total_consumers_audited: `15`
- raw_only_high_priority_paths: `0`
- already_support_both: `12`
- csi_value_weakened_by_unified_gaps: `False`

## Summary
1. raw-only key paths: `none`
2. already support ExtractedCSI: `src/beamforming/utils/sionna_native_beamforming_chain.py::compute_project_precoder_per_subcarrier, src/beamforming/utils/sionna_native_learned_beamforming.py::infer_learned_precoder, scripts/sionna_csi_backed_beamforming_chain.py::main, scripts/sionna_native_channel_beamforming_chain.py::main, scripts/sionna_native_ofdm_learned_beamforming_chain.py::main, scripts/validate_csi_same_batch_equivalence.py::main, scripts/audit_sionna_csi_interface.py::main, scripts/benchmark_sionna_extracted_h_consistency.py::main, scripts/sionna_extract_channel_hf_demo.py::main, scripts/run_sionna_native_learned_chain_minibench.py::main, tests/test_csi_interface.py::CSI base tests, README.md::command examples and branch status, docs/sionna_native_channel_extraction.md::CSI interface documentation`
3. priority migration targets: `src/beamforming/utils/sionna_native_beamforming_chain.py::compute_project_precoder_per_subcarrier, src/beamforming/utils/sionna_native_learned_beamforming.py::infer_learned_precoder, scripts/sionna_csi_backed_beamforming_chain.py::main, scripts/sionna_native_channel_beamforming_chain.py::main`
4. ununified consumers still weaken v0.6.0 CSI value: `False`

| File | Function/Script | Input | Should support CSI | Priority | Recommended change |
| --- | --- | --- | --- | --- | --- |
| src/beamforming/utils/sionna_native_beamforming_chain.py | compute_project_precoder_per_subcarrier | both | True | high | Use as_project_h_f() internally so analytic precoders accept ExtractedCSI and raw H_f. |
| src/beamforming/utils/sionna_native_learned_beamforming.py | infer_learned_precoder | both | True | high | Normalize inputs through as_project_h_f() and record CSI provenance in inference metadata. |
| scripts/sionna_csi_backed_beamforming_chain.py | main | both | True | high | Pass ExtractedCSI directly into analytic and learned consumers; keep summary fields for provenance. |
| scripts/sionna_native_channel_beamforming_chain.py | main | both | True | high | Prefer context.csi when available and only keep raw H_f as fallback for legacy path. |
| scripts/sionna_native_ofdm_learned_beamforming_chain.py | main | both | True | medium | CSI-backed path is wired in; keep summary provenance fields and raw fallback for backward compatibility. |
| scripts/validate_csi_same_batch_equivalence.py | main | both | True | medium | Use one shared ExtractedCSI object and compare raw fallback vs CSI-backed consumers under one realization. |
| scripts/audit_sionna_csi_interface.py | main | both | True | medium | Audit consumer calls using ExtractedCSI directly and summarize provenance completeness. |
| scripts/benchmark_sionna_extracted_h_consistency.py | main | both | True | medium | Benchmark now prefers context.csi when available; preserve quick/limited wording and raw fallback. |
| scripts/sionna_extract_channel_hf_demo.py | main | both | True | medium | Prefer CSI summary in artifacts instead of direct tensor-only reporting. |
| scripts/run_sionna_native_learned_chain_minibench.py | main | both | True | low | Keep optional minibench aligned with ExtractedCSI-first input summary while preserving raw fallback. |
| scripts/sionna_native_ofdm_beamforming_chain.py | main | raw_h_f | False | low | Legacy v0.4-era project-assisted script; optional future cleanup if this path is revived. |
| tests/test_csi_interface.py | CSI base tests | ExtractedCSI | True | low | Keep schema tests and extend adapters in dedicated adapter test file. |
| tests/test_sionna_channel_extraction_optional.py | optional Sionna script tests | unknown | True | medium | Cover audit/demo/comparison scripts that should now surface CSI-backed behavior. |
| README.md | command examples and branch status | both | True | medium | Document ExtractedCSI-first path and raw H_f fallback boundaries. |
| docs/sionna_native_channel_extraction.md | CSI interface documentation | both | True | medium | Add consumer-unification status table and clarify remaining raw fallback consumers. |
