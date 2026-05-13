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

This document now tracks the published `v0.5.0` and `v0.6.0` states together with the `v0.7.0` candidate branch for CSI consumer unification on top of the optional Sionna-native channel-extraction bridge.

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

- this script is now explicitly labeled `comparison_type=cross_run_comparison`
- `not_strict_equivalence_test=true` for this artifact
- the mismatch is now audited as `cross_run_comparison_without_shared_realization`
- the goal of this phase is interface/schema/provenance hardening
- it is not a new claim that the CSI-backed path materially improves metrics
- it also does not change the current benchmark boundary

## CSI Same-batch Equivalence

The new same-batch validation fixes the earlier ambiguity by forcing the raw extracted-H path and the CSI-backed path to reuse the exact same:

- Sionna channel tensor
- extracted `H_f`
- `ExtractedCSI` object
- bits and mapped symbols
- receiver configuration
- noise configuration

Current result:

- `same_channel_tensor_used = true`
- `same_bits_used = true`
- `same_noise_config_used = true`
- `same_receiver_config_used = true`
- `numeric_consistency_within_tolerance = true`
- `ranking_consistent = true`
- `max_abs_diff_sum_rate = 0.0`
- `max_abs_diff_symbol_mse = 0.0`
- `max_abs_diff_sinr_db = 0.0`

This is the correct interpretation boundary for the current interface work:

- the CSI-backed interface is numerically consistent with the raw extracted-H path under a shared realization
- this does not imply that independent reruns should be expected to match numerically
- this still does not change the branch boundary to full native-only

## Previous Mismatch Root Cause

The mismatch audit now reports:

- `comparison_independent_runs = true`
- `same_seed_used = true`
- `same_channel_tensor_shape_metadata = true`
- `same_selected_ofdm_symbol = true`
- `same_effective_subcarrier_indices = true`
- `same_receiver_mode = true`
- `same_bits_used = false`
- `same_symbols_used = false`
- `same_noise_realization_used = false`
- `csi_interface_bug_evidence = false`

So the earlier numeric inconsistency was primarily caused by comparing separate runs without a shared realization fixture, not by a confirmed CSI-interface bug.

The supported wording remains:

- cross-run comparison is not a strict equivalence test
- same-batch equivalence is the valid place to claim numerical consistency
- CSI interface improves provenance clarity and deterministic reuse under shared realization
- native-channel-assisted plus native-receiver-assisted
- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

## v0.6.0 Candidate Status

Compact CSI result table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| CSI audit | `passed` | `ExtractedCSI` provenance is complete enough for current project and learned consumers |
| CSI-backed beamforming | `native_receiver_success=true` | CSI-backed path enters native receiver chain successfully |
| same-batch equivalence | `passed` | raw extracted-H and CSI-backed paths are numerically consistent under one shared realization |
| previous mismatch root cause | `cross_run_comparison_without_shared_realization` | earlier mismatch was cross-run comparison, not CSI-interface bug evidence |

The current supported summary is:

- same-batch equivalence passed
- previous mismatch was cross-run comparison
- CSI interface improves provenance and deterministic reuse
- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

## PrecoderOutput Interface Motivation

The `v0.7.0` consumer-unification phase made `ExtractedCSI` the preferred input interface for high-priority CSI consumers, but the project still passed raw `F_f=(B,Nsc,Nt,K)` tensors directly across analytic precoders, learned beamformers, and native receiver scripts.

The next interface-hardening step is to standardize that output side as well so that:

- analytic project precoders emit one validated `PrecoderOutput` schema
- learned beamformer inference emits the same validated schema without losing checkpoint or teacher provenance
- native receiver paths consume a reusable container instead of ad hoc raw tensor wiring
- raw `F_f` remains available as a backward-compatible fallback where older scripts still expect it

## PrecoderOutput Schema

Current normalized fields:

- `f_f`: complex torch tensor with shape `(B,Nsc,Nt,K)`
- `source`: one of `project_rzf`, `project_wmmse_iter_5`, `learned_residual_rzf`, `learned_residual_wmmse_distill`, `sionna_rzf_future`
- `method`
- `input_csi_summary`
- `axes = {B:0, Nsc:1, Nt:2, K:3}`
- `shape = {B, Nsc, Nt, K}`
- `num_users`
- `num_bs_ant`
- `num_subcarriers`
- `power_normalized`
- `power_norm`
- `teacher_used_during_inference`
- `project_side_precoder`
- `sionna_native_precoder`
- `full_native_only`
- `metadata`

Current provenance metadata includes:

- `input_csi_source`
- `input_h_f_shape`
- `checkpoint_path`
- `skipped_missing_checkpoint`
- `teacher_used_during_inference`
- `fallback_reason`

## CSI + PrecoderOutput Unified Flow

The current preferred bridge now becomes:

- `ExtractedCSI -> PrecoderOutput -> native receiver path`

This means the mainline Sionna-assisted workflow can now keep:

- one shared CSI object as the preferred `H_f` input
- one standardized precoder container as the preferred `F_f` output
- one native receiver bridge that consumes either container-first interfaces or raw fallbacks when explicitly requested

## Current PrecoderOutput Status

Current supported summary:

- analytic `project_rzf` and `project_wmmse_iter_5` now support `return_precoder_output=True`
- learned `learned_residual_rzf` and `learned_residual_wmmse_distill` now support `return_precoder_output=True`
- learned `PrecoderOutput` artifacts explicitly preserve `teacher_used_during_inference=false`
- the native receiver bridge accepts `PrecoderOutput` directly
- `ExtractedCSI` is the preferred input interface and `PrecoderOutput` is the preferred output interface
- raw `H_f` and raw `F_f` remain backward-compatible fallbacks
- raw `F_f` remains a backward-compatible fallback
- same-batch raw-`F_f` vs `PrecoderOutput` validation now reuses one shared CSI object, one shared raw `F_f` per method, one shared bit/symbol batch, one shared noise realization, and one shared native receiver configuration
- the old raw-`F_f` vs `PrecoderOutput` ranking mismatch is now explicitly explained as a cross-run comparison artifact rather than `PrecoderOutput` bug evidence
- this still does not change the benchmark boundary to full native-only

Compact PrecoderOutput table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| PrecoderOutput schema | `implemented` | standardized project-side `F_f=(B,Nsc,Nt,K)` output container |
| analytic methods emit PrecoderOutput | `supported` | project `RZF/WMMSE` outputs now have a reusable bridge format |
| learned methods emit PrecoderOutput | `supported` | learned residual methods keep checkpoint and teacher provenance in one object |
| native receiver consumes PrecoderOutput | `supported` | receiver path accepts standardized output object directly |
| raw F_f fallback | `retained` | older scripts/tests can still use legacy raw tensors |
| same-batch raw-vs-PrecoderOutput equivalence | `passed` | shared-realization validation gives exact metric agreement within tolerance |
| previous raw-vs-PrecoderOutput mismatch root cause | `cross_run_comparison_without_shared_csi_and_precoder_realization` | prior mismatch came from independent reruns |
| strict raw-vs-PrecoderOutput equivalence claim on cross-run artifact | `false` | only the new same-batch validation can justify the strict claim |
| current v0.8.0 candidate status | `release hardening` | manifest, minimal reproduction, release notes, and PR text are prepared for the bridge |

PrecoderOutput same-batch interpretation:

- `PrecoderOutput.f_f` is numerically identical to the corresponding raw `F_f` under one shared realization
- native receiver metrics are numerically consistent under one shared CSI / `F_f` / symbol / noise / receiver-config fixture
- the old cross-run comparison remains useful for artifact-level auditing, but it is not a strict equivalence test
- no new fallback is introduced by this bridge
- this still does not change the boundary to full native-only

Boundary remains:

- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

## Native Precoder API Probe Status

After `v0.8.0`, the next interface-hardening question is not "replace the project-side precoder now", but:

- can the installed Sionna native precoder API be called reproducibly?
- how far is its tensor contract from the current `ExtractedCSI` / `PrecoderOutput` contract?
- can one adapter bridge prove shape and receiver compatibility without overstating integration status?

Current native-precoder probe status:

- `RZFPrecoder` is available in Sionna `2.0.1`
- it expects:
  - `x = (B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size)`
  - `h = (B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size)`
- this is not a direct drop-in replacement for project `H_f=(B,Nsc,K,Nt)`
- the current branch adds an optional adapter:
  - `ExtractedCSI -> Sionna native probe inputs`
  - `Sionna native output -> PrecoderOutput`
- current supported interpretation remains:
  - callable native API
  - partial bridge compatibility
  - current minimal probe can convert native RZF output to `PrecoderOutput` and enter the current native receiver path
  - no direct mainline replacement yet

Current recommendation:

- keep `project_rzf` as the clean mainline precoder path
- treat `RZFPrecoder` as an optional native reference path behind explicit adapter logic
- use the bridge result to map future shape / stream-management integration cost
- same-realization validation currently supports `sionna_rzf_precoder` as an optional method because:
  - one shared `ExtractedCSI` object is reused
  - the converted native output enters the native receiver path successfully
  - semantic compatibility passes under the shared realization
  - but strict numerical equivalence still does not pass

Current semantic-alignment result:

- same-realization comparison:
  - `relationship_status = close_but_different`
  - `semantic_compatibility_passed = true`
  - `strict_equivalence_claim_allowed = false`
  - `max_abs_diff_f_f_if_comparable = 0.061414435505867004`
  - `abs_diff_sum_rate = 0.09229850769042969`
  - `abs_diff_symbol_mse = 0.0006549134850502014`
  - `abs_diff_sinr_db = 0.07219910621643066`
- quick seed/SNR sweep (`seeds=1,2,3`, `snr=0,5,10,15,20 dB`):
  - `RZFPrecoder` is callable on all evaluated rows
  - conversion to `PrecoderOutput` succeeds on all evaluated rows
  - native receiver success is true on all evaluated rows
  - all rows remain `close_but_different`
  - this supports optional-native-method integration, not a strict-equivalence claim

Compact native precoder table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| `sionna_rzf_available` | `true` | current Sionna install exposes `RZFPrecoder` |
| `sionna_rzf_callable` | `true` | minimal native precoder call works |
| `converted_to_precoder_output` | `true` | native output can be bridged back to project `PrecoderOutput` |
| `native_receiver_success` | `true` | optional native method enters the receiver path |
| `relationship_status` | `close_but_different` | semantic alignment passes without strict value match |
| `strict_equivalence_claim_allowed` | `false` | do not claim strict equivalence to `project_rzf` |
| `full_native_only` | `false` | benchmark boundary remains non-native-only |

Contract-aware interpretation:

- the optional native method now has an explicit contract schema
- the schema hardens:
  - expected native input/output tensor layouts
  - adapter-required alignment to `ExtractedCSI` and `PrecoderOutput`
  - skip/fallback policy
  - comparison semantics
- the successful contract-aware path still remains:
  - optional method bridge
  - `close_but_different`
  - not strict equivalent
  - not full native-only benchmark

## v1.0.0-rc1 Interface-first Overview

Current RC overview:

- `Sionna OFDMChannel -> ExtractedCSI -> PrecoderOutput -> native receiver path`
- channel extraction, CSI interface, CSI consumers, precoder output, optional native precoder bridge, and contract hardening are now documented as one interface-first stack
- this is still a release candidate for interfaces and reproducibility, not a production or full-native benchmark claim

Current stable release overview:

- `v1.0.0` keeps the same interface-first chain
- stable finalization adds a stable artifact manifest and a stable minimal reproduction wrapper
- the boundary still remains non-full-native and non-production

Post-`v1.0.0` maintenance:

- release body consistency audit
- artifact reproducibility audit
- optional Sionna regression monitor
- maintenance issue list in [docs/maintenance/post_v1_maintenance_issues.md](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/docs/maintenance/post_v1_maintenance_issues.md)

Boundary remains unchanged:

- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

Current consumer-unification validation commands:

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
python scripts/validate_precoder_output_same_batch_equivalence.py \
  --out outputs/sionna_channel_extraction/precoder_output_same_batch_equivalence.json

python scripts/audit_precoder_output_comparison_mismatch.py \
  --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv \
  --out outputs/sionna_channel_extraction

python scripts/compare_raw_ff_vs_precoder_output.py \
  --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv \
  --out outputs/sionna_channel_extraction

python scripts/generate_sionna_precoder_interface_artifact_manifest.py \
  --out outputs/sionna_channel_extraction/precoder_interface_artifact_manifest.json

python scripts/reproduce_sionna_precoder_interface_minimal.py \
  --out outputs/repro/sionna_precoder_interface_minimal_summary.json
```

## CSI Consumer Unification Status

The next interface-hardening step is to make `ExtractedCSI` the preferred input for all important `H_f=(B,Nsc,K,Nt)` consumers while keeping raw `H_f` as a backward-compatible fallback.

Current audited state:

- analytic precoders now accept `ExtractedCSI`, raw `H_f`, or dict inputs containing `h_f`
- learned beamformer inference now accepts `ExtractedCSI` directly and records provenance-oriented input metadata
- CSI-backed and native-channel-assisted beamforming scripts now summarize `input_type`, `csi_interface_used`, `project_h_f_assisted`, `extracted_h_f_used`, and `full_native_only`
- older benchmark/minibench paths may still keep raw-`H_f` fallback behavior, but the preferred path is CSI-backed

Compact unification table:

| Consumer group | Current support | Notes |
| --- | --- | --- |
| analytic precoder consumers | `ExtractedCSI + raw fallback` | `compute_project_precoder_per_subcarrier(...)` normalizes inputs through `as_project_h_f(...)` |
| learned beamformer consumers | `ExtractedCSI + raw fallback` | `infer_learned_precoder(...)` accepts CSI directly and keeps `teacher_used_during_inference=false` |
| native receiver chain scripts | `CSI-first where available` | summaries now report CSI provenance fields explicitly |
| comparison / benchmark scripts | `CSI-first with fallback retained` | same-batch equivalence remains the only valid strict-equivalence claim path |
| docs / README command examples | `CSI-backed path preferred` | raw tensor path remains documented only as fallback / legacy compatibility |

Current audit headline:

- `total_consumers_audited = 15`
- `raw_only_high_priority_paths = 0`
- `already_support_both = 12`
- `csi_value_weakened_by_unified_gaps = false`

Current unified-consumer demo result:

- `status = ok`
- `csi_object_created = true`
- `same_csi_object_used_for_all_methods = true`
- `all_consumers_accept_csi = true`
- `native_receiver_success = true`
- `teacher_used_during_inference = false`
- `failed_consumers = []`
- `no_new_fallback_introduced = true`

Current unified-vs-baseline interpretation:

- `comparison_type = cross_run_comparison`
- `same_seed_used = true`
- `same_csi_tensor_signature = false`
- `strict_equivalence_claim_allowed = false`
- the current unified-vs-baseline artifact should not be read as a strict equivalence claim
- strict shared-realization equivalence remains the earlier same-batch validation

The supported interpretation remains unchanged:

- CSI consumer unification improves provenance clarity and deterministic reuse
- unified-vs-baseline reruns are currently cross-run comparisons and may differ numerically unless they share one realization fixture
- it does not introduce a new claim of full native-only benchmarking
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

## v0.7.0 Candidate Status

Compact consumer-unification table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| total_consumers_audited | `15` | audit covers analytic, learned, native-chain, comparison, tests, and docs paths |
| raw_only_high_priority_paths | `0` | no key consumer remains blocked on raw-only `H_f` |
| already_support_both | `12` | most important consumers now accept `ExtractedCSI` and keep raw fallback |
| all_consumers_accept_csi | `true` | unified demo shows current key paths accept one shared `ExtractedCSI` object |
| no_new_fallback_introduced | `true` | unification did not add new fallback behavior |
| unified-vs-baseline strict equivalence allowed | `false` | current comparison remains cross-run only |

The current supported summary is:

- `ExtractedCSI` is now the preferred input interface for current key consumers
- raw `H_f` remains a backward-compatible fallback
- high-priority raw-only gaps are zero
- unified-vs-baseline is a cross-run comparison, not a strict equivalence test
- provenance clarity is improved without introducing new fallback
- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only
