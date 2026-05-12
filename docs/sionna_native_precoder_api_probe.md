# Sionna Native Precoder API Probe

After `v0.8.0`, the project already has:

- preferred CSI input interface: `ExtractedCSI`
- preferred precoder output interface: `PrecoderOutput`
- same-batch raw-`F_f` vs `PrecoderOutput` equivalence validation

The next question is narrower: how close is the currently installed Sionna 2.0.1 native precoder API to the project's interface contract?

This phase is an API/compatibility probe only. It is **not** a claim that the project has replaced the project-side precoder path with a fully native Sionna precoder path.

## Why audit the native precoder API now

The current project-side bridge already proves:

- `ExtractedCSI` can carry provenance-aware `H_f=(B,Nsc,K,Nt)`
- `PrecoderOutput` can carry provenance-aware `F_f=(B,Nsc,Nt,K)`
- the native receiver path can consume `PrecoderOutput`

The remaining gap is the native precoder API contract itself:

- expected Sionna input shapes
- expected Sionna channel tensor layout
- how `StreamManagement` constrains valid calls
- whether Sionna output can be converted back to a `PrecoderOutput`

## Sionna RZFPrecoder audit result

Current observed result on Sionna `2.0.1`:

- `sionna.phy.ofdm.RZFPrecoder` is available
- constructor requires:
  - `resource_grid`
  - `stream_management`
- call signature is:
  - `call(x, h, alpha=0.0)`

Current documented native contract:

- `x` shape:
  - `(B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size)`
- `h` shape:
  - `(B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size)`
- `x_precoded` output shape:
  - `(B, num_tx, num_tx_ant, num_ofdm_symbols, fft_size)`

This is not the same contract as the project's mainline:

- project CSI:
  - `H_f = (B,Nsc,K,Nt)`
- project precoder:
  - `F_f = (B,Nsc,Nt,K)`

So the current result is:

- native API is real and callable
- direct drop-in substitution is **not** clean
- current best path is an adapter bridge, not a direct mainline replacement

## Shape mapping used in the current probe

Current probe-only mapping:

- `ExtractedCSI -> Sionna precoder input`
  - treat `num_tx = 1`
  - treat `num_streams_per_tx = K`
  - build one minimal `ResourceGrid`
  - build `StreamManagement(rx_tx_association=ones(K,1), num_streams_per_tx=K)`
  - expand the extracted channel into the higher-rank Sionna channel layout required by `RZFPrecoder`

- `Sionna precoder output -> PrecoderOutput`
  - take `x_precoded[:,0,...]`
  - permute axes to project order
  - convert to `F_f=(B,Nsc,Nt,K)`
  - apply project-side power normalization before storing as `PrecoderOutput`

This means the current adapter is:

- a compatibility bridge
- auditable
- reusable for probe/demo work

but still not a proof that the project has eliminated the project-side bridge boundary.

## Probe result

Current probe summary fields:

- `sionna_rzf_available`
- `sionna_rzf_callable`
- `extracted_csi_used`
- `sionna_precoder_success`
- `sionna_output_shape`
- `converted_to_precoder_output`
- `project_rzf_output_shape`
- `shape_compatible`
- `power_norm_project`
- `power_norm_sionna`
- `max_abs_diff_if_comparable`
- `native_receiver_success_if_attempted`
- `fallback_used`
- `fallback_reason`
- `recommended_next_step`

Current supported interpretation:

- if `sionna_rzf_callable=true`, the installed API is usable for minimal native probing
- if `converted_to_precoder_output=true`, the adapter bridge works for the current shape contract
- if `native_receiver_success_if_attempted=true`, the converted native output can traverse the current receiver bridge

That still does **not** imply:

- full native-only benchmark
- Sionna RT
- ray tracing
- 5G NR full stack

## Comparison against project_rzf

Current comparison target is deliberately narrow:

- shape compatibility
- power normalization compatibility
- receiver-path usability
- coarse metric closeness

It is not framed as a new performance claim.

Recommended interpretation:

- `project_rzf` remains the clean mainline
- `RZFPrecoder` can now be treated as an optional native reference path
- the current branch reduces uncertainty around native-precoder integration cost
- the current branch does not yet remove the project-side precoder bridge limitation

## Current boundary

The supported boundary remains:

- optional dependency only
- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- no stable learned `> WMMSE-iter5` claim
