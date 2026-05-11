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

## Native-Channel-Assisted Beamforming Outcome

Current native-channel-assisted chain status:

- `extraction_success = true`
- `project_h_f_assisted = false`
- `native_receiver_success = true`
- learned `residual_rzf` can run with extracted `H_f`
- learned `residual_wmmse_distill` can also run with extracted `H_f`
- `teacher_used_during_inference = false` remains preserved

## Remaining Boundary

This shrinks the earlier `project-H_f-assisted` limitation, but it still does not justify calling the system a full native-only benchmark:

- the receiver path is native Sionna
- the channel tensor now comes from a native Sionna path
- but the project precoder and learned-model interface remain project-side components layered on top of that extraction bridge
- therefore the correct description is:
  native-channel-assisted and native-receiver-assisted,
  but not yet a full native-only benchmark
