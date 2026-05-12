# Sionna Native Channel Extraction

This phase extends the `v0.4.0` native receiver work toward a narrower goal: reduce the current `project-H_f-assisted` limitation by extracting a project-compatible frequency-domain channel tensor directly from Sionna channel outputs where possible.

## Starting Point

Current `v0.4.0` native-chain boundary:

- real Sionna receiver path
- learned `residual_rzf` / `residual_wmmse_distill` enter the native receiver path
- `teacher_used_during_inference = false`
- `H_f` / precoder side remains project-assisted
- not a full native-only benchmark

## Goal

Try to extract or construct:

- `H_f = (B, Nsc, K, Nt)`

from Sionna-native channel tensors so the existing project precoder and learned-beamformer interfaces can consume a less assisted channel representation.

## Current Status

This document now tracks the published `v0.5.0` state together with the follow-on CSI-interface standardization branch for the optional Sionna-native channel-extraction bridge.

Compact result table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| channel tensor shape | `[8,4,1,1,16,2,19]` | observed native Sionna channel tensor layout |
| extracted `H_f` shape | `[8,16,4,16]` | project-compatible `H_f=(B,Nsc,K,Nt)` |
| axis validation | `spot_check_max_abs_diff=0.0` | bridge axes/data-symbol selection are consistent |
| native receiver success | `true` | supported path is native-channel-assisted + native-receiver-assisted |
| proxy/native rank agreement | `0.226667` | proxy metric cannot replace native receiver metric |

## What This Phase Audits

- `OFDMChannel`
- `ApplyOFDMChannel`
- `RayleighBlockFading`
- `GenerateOFDMChannel`, if available
- `cir_to_ofdm_channel`, if available
- `subcarrier_frequencies`, if available

## Current Reading Rules

- successful `H_f` extraction reduces the `project-H_f-assisted` limitation
- it does not automatically imply a full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only
- not production e2e

## Current Audit Result

Current `Sionna 2.0.1` finding:

- `OFDMChannel(return_channel=True)` does return a channel tensor
- the observed tensor shape is:
  `h = [B, rx, rx_ant, tx, tx_ant, ofdm_symbol, fft_bin]`
- for the current MU downlink bridge this maps to:
  `rx -> user K`
  `tx_ant -> base-station antenna Nt`
  `ofdm_symbol -> choose a data symbol`
  `fft_bin -> select effective subcarriers`

Under the current assumptions, the bridge can convert that tensor to:

- `H_f = (B, Nsc, K, Nt)`

## Current Extraction Outcome

The current demo successfully extracts:

- `sionna_channel_tensor_shape = [8, 4, 1, 1, 16, 2, 19]`
- `extracted_h_f_shape = [8, 16, 4, 16]`
- `project_h_f_shape_compatible = true`

This means the project-side `H_f` interface can now be fed from a real Sionna channel tensor in the current synthetic setup.

## Axis Validation and Data-Symbol Sanity

The current axis-validation script verifies that the bridge is not silently selecting the wrong OFDM symbol or wrong tensor axes.

Current result:

- `hf_axis_validation.json` passes with:
  - `axis_spot_check_passed = true`
  - `spot_check_max_abs_diff = 0.0`
  - `selected_data_symbol_indices = [1]`
  - `effective_subcarrier_count_matches_nsc = true`
- in the current pilot-aware grid, OFDM symbol `0` is pilot-bearing and OFDM symbol `1` is the first data-bearing symbol
- this means the extracted bridge should read the current default as:
  - `selected_ofdm_symbol = first_data -> actual index 1`
  - `effective_subcarriers = all_effective`

For the current `num_tx=1`, `rx_ant=1` downlink bridge, the hidden squeeze/transpose risk is now considered low, but it would need to be re-audited if future work adds multi-TX or multi-RX-antenna semantics.

## Native-Channel-Assisted Beamforming Outcome

Current native-channel-assisted chain status:

- `extraction_success = true`
- `project_h_f_assisted = false`
- `native_receiver_success = true`
- learned `residual_rzf` can run with extracted `H_f`
- learned `residual_wmmse_distill` can also run with extracted `H_f`
- `teacher_used_during_inference = false` remains preserved

The current native-channel-assisted beamforming path now uses one shared Sionna channel realization for both:

- extracted `H_f` used by the project/learned precoder interface
- `h_full` used by the native Sionna receiver path

This removes the earlier consistency risk where the precoder and receiver could have been driven by different channel realizations.

## Extracted-H Consistency Benchmark

The quick consistency benchmark currently runs:

- seeds: `1,2,3`
- SNR: `0,5,10,15,20 dB`
- quick batch size

Current quick result:

- `extraction_success = true` on all evaluated rows
- `native_receiver_success = true` on all evaluated rows
- proxy/native exact rank agreement is only `0.226667`

This matters for interpretation:

- extracted `H_f` is stable and usable
- but project-side proxy ranking is not a strong substitute for the native receiver ranking under the extracted-H path

Current quick learned interpretation:

- `learned_residual_rzf` vs `project_rzf` mean gap: `-1.681560%`
- `learned_residual_rzf` vs `project_wmmse_iter_5` mean gap: `+4.735929%`

This is still a quick consistency study only. It is not enough to support a stable claim that learned methods beat `WMMSE-iter5`.

## Extraction Config Sweep

The extraction-config sweep now checks:

- `selected_ofdm_symbol`: `first_data`, `last_data`, `all_data_average`
- `effective_subcarriers`: `all_effective`, `center_8`, `center_16`
- `normalize_channel`: `true/false`

Current quick sweep result:

- extraction succeeds across all currently tested settings
- rank remains stable (`rank_mean = 4.0` in the current setup)
- normalization behaves as expected by forcing per-subcarrier Frobenius norm near `1.0`

Current recommended default remains:

- `selected_ofdm_symbol = first_data`
- `effective_subcarriers = all_effective`
- `normalize_channel = false`

## Project-H_f-Assisted vs Extracted-H_f Path

The comparison against the earlier `v0.4.0` project-assisted path now shows:

- the extracted-H path changes the single-run method ranking
- `learned_residual_rzf` remains close to analytic baselines on both paths
- `project_wmmse_iter_5` remains a strong analytic baseline
- the `project-H_f-assisted` limitation is genuinely reduced because `H_f` now comes from a real Sionna channel tensor

But the branch should still not be described as a full native-only benchmark:

- the receiver path is real Sionna
- the channel extraction path is now real Sionna
- the precoder and learned-beamformer interface remain project-side bridge components
- therefore the correct label remains:
  native-channel-assisted and native-receiver-assisted,
  not full native-only

## Remaining Boundary

This shrinks the earlier `project-H_f-assisted` limitation, but it still does not justify calling the system a full native-only benchmark:

- the receiver path is native Sionna
- the channel tensor now comes from a native Sionna path
- but the project precoder and learned-model interface remain project-side components layered on top of that extraction bridge
- therefore the correct description is:
  native-channel-assisted and native-receiver-assisted,
  but not yet a full native-only benchmark

## Hard Boundaries

- extracted-H_f reduces the `project-H_f-assisted` limitation
- current supported description is `native-channel-assisted + native-receiver-assisted`
- not full native-only benchmark
- proxy metric cannot replace native receiver metric
- no stable learned `> WMMSE-iter5` claim
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

## CSI Interface Motivation

The `v0.5.0` extraction bridge proved that a native Sionna channel tensor can be converted into project-side `H_f=(B,Nsc,K,Nt)`, but the earlier consumers still relied on ad hoc transpose/squeeze handling.

The current follow-on branch introduces a standardized `ExtractedCSI` object so that:

- project precoders consume one validated `H_f` shape
- learned beamformers consume the same validated `H_f` shape
- native receiver experiments keep explicit provenance for where that `H_f` came from
- future DeepMIMO or other CSI sources can conform to the same container without reusing Sionna-specific bridge code

## ExtractedCSI Schema

Current normalized fields:

- `h_f`: complex torch tensor with shape `(B,Nsc,K,Nt)`
- `source`: one of `sionna_ofdm_channel`, `synthetic_project`, `deepmimo_future`
- `source_component`
- `axes = {B:0, Nsc:1, K:2, Nt:3}`
- `shape = {B, Nsc, K, Nt}`
- `selected_ofdm_symbol`
- `effective_subcarrier_indices`
- `num_users`
- `num_bs_ant`
- `num_subcarriers`
- `project_h_f_assisted`
- `extracted_h_f_used`
- `full_native_only`
- `metadata`

Key provenance metadata currently recorded for the Sionna path:

- `original_sionna_h_shape`
- `original_axes`
- `selected_data_symbol`
- `selected_data_symbol_indices`
- `pilot_symbol_indices`
- `effective_subcarrier_ind`
- `extraction_success`
- `fallback_reason`
- `conversion_meta`

## CSI Provenance Audit Result

The CSI-interface audit currently reports:

- `csi_interface_used = true`
- `h_f_shape_ok = true`
- `axes_metadata_complete = true`
- `original_sionna_h_shape_present = true`
- `selected_data_symbol_not_pilot = true`
- `effective_subcarrier_count_matches_nsc = true`
- `project_h_f_assisted = false`
- `extracted_h_f_used = true`
- `full_native_only = false`
- `project_rzf_consumes_csi = true`
- `learned_residual_rzf_consumes_csi = true`

This confirms that the standardized CSI object is usable by both the analytic project precoder path and the learned residual-RZF path while keeping the same boundary interpretation as `v0.5.0`.

## CSI-backed Beamforming Result

The CSI-backed beamforming chain now runs the extracted-channel path through the standardized `ExtractedCSI` object before the existing project and learned precoder interfaces.

Current single-run result:

- `csi_interface_used = true`
- `csi_source = sionna_ofdm_channel`
- `project_h_f_assisted = false`
- `extracted_h_f_used = true`
- `full_native_only = false`
- `native_receiver_success = true`
- learned methods keep `teacher_used_during_inference = false`

Current methods evaluated:

- `project_rzf`
- `project_wmmse_iter_5`
- `learned_residual_rzf`
- `learned_residual_wmmse_distill`

## Raw Extracted-H vs CSI-backed Comparison

The current comparison between the earlier raw extracted-H script path and the CSI-backed path reports:

- no new fallback was introduced
- provenance clarity is improved because the CSI-backed path stores a reusable validated summary object
- the current separate reruns are not numerically identical and do not preserve exact ranking

This should be interpreted carefully:

- the goal of this phase is interface/schema/provenance hardening
- it is not a new claim that the CSI-backed path materially improves metrics
- it also does not change the current benchmark boundary

The supported wording remains:

- native-channel-assisted plus native-receiver-assisted
- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only
