# Precoder Interface Audit

- total_consumers_audited: `14`
- raw_only_high_priority_paths: `0`
- already_support_both: `9`
- learned_teacher_flag_all_false: `True`

## Summary
1. raw-only key paths: `scripts/sionna_native_channel_beamforming_chain.py::main`
2. already support PrecoderOutput: `src/beamforming/utils/sionna_native_beamforming_chain.py::compute_project_precoder_per_subcarrier, src/beamforming/utils/sionna_native_beamforming_chain.py::apply_project_precoder_to_sionna_grid, src/beamforming/utils/sionna_native_learned_beamforming.py::infer_learned_precoder, src/beamforming/utils/sionna_native_learned_beamforming.py::run_native_receiver_with_precoder, scripts/sionna_csi_backed_beamforming_chain.py::main, scripts/demo_unified_csi_consumers.py::main, scripts/sionna_native_ofdm_learned_beamforming_chain.py::main, scripts/demo_unified_csi_and_precoder_interfaces.py::main, tests/test_precoder_interface.py::schema tests, tests/test_precoder_input_adapters.py::adapter tests, README.md::Sionna optional docs, docs/sionna_native_channel_extraction.md::CSI/precoder bridge docs`
3. high-priority raw-only gap exists: `False`
4. learned outputs track teacher_used_during_inference=false: `True`

| File | Function/Script | Output | Input | Priority | Recommended change |
| --- | --- | --- | --- | --- | --- |
| src/beamforming/utils/sionna_native_beamforming_chain.py | compute_project_precoder_per_subcarrier | both | both | high | Prefer return_precoder_output=True in new scripts while preserving raw F_f fallback. |
| src/beamforming/utils/sionna_native_beamforming_chain.py | apply_project_precoder_to_sionna_grid | unknown | both | high | Consume PrecoderOutput via as_project_f_f() and keep raw tensor compatibility. |
| src/beamforming/utils/sionna_native_learned_beamforming.py | infer_learned_precoder | both | both | high | Emit PrecoderOutput for learned inference and preserve teacher_used_during_inference=false provenance. |
| src/beamforming/utils/sionna_native_learned_beamforming.py | run_native_receiver_with_precoder | unknown | both | high | Prefer PrecoderOutput in receiver-chain entrypoints and surface interface metadata in rows. |
| scripts/sionna_csi_backed_beamforming_chain.py | main | both | both | high | Default to PrecoderOutput and keep --raw-f-f fallback for compatibility comparisons. |
| scripts/demo_unified_csi_consumers.py | main | PrecoderOutput | PrecoderOutput | high | Use one shared ExtractedCSI object and require all evaluated methods to emit PrecoderOutput. |
| scripts/sionna_native_ofdm_learned_beamforming_chain.py | main | both | both | medium | Record precoder interface provenance for analytic and learned paths while preserving raw F_f fallback semantics. |
| scripts/sionna_native_channel_beamforming_chain.py | main | raw_f_f | raw_f_f | medium | Optional future migration to PrecoderOutput if this legacy extracted-H script remains a maintained path. |
| scripts/validate_csi_same_batch_equivalence.py | main | raw_f_f | raw_f_f | low | Keep same-batch equivalence focused on raw-vs-CSI H_f path unless a dedicated raw-vs-PrecoderOutput same-batch test is added. |
| scripts/demo_unified_csi_and_precoder_interfaces.py | main | PrecoderOutput | PrecoderOutput | high | Serve as the main unified demo for ExtractedCSI -> PrecoderOutput -> native receiver flow. |
| tests/test_precoder_interface.py | schema tests | PrecoderOutput | PrecoderOutput | low | Keep schema and provenance validation independent from optional Sionna runtime. |
| tests/test_precoder_input_adapters.py | adapter tests | both | both | low | Preserve raw tensor fallback while keeping PrecoderOutput as the preferred interface. |
| README.md | Sionna optional docs | both | both | medium | Document PrecoderOutput as preferred output while keeping raw F_f as backward-compatible fallback. |
| docs/sionna_native_channel_extraction.md | CSI/precoder bridge docs | both | both | medium | Add ExtractedCSI -> PrecoderOutput -> native receiver flow and cross-run comparison caveat. |
