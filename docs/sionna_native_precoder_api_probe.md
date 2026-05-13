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

## Same-realization semantic alignment

The next validation step is stricter than the earlier probe:

- reuse one `ExtractedCSI` object
- reuse one shared symbol batch
- reuse one shared native receiver configuration
- reuse one shared noise configuration

Current same-realization result:

- `sionna_rzf_available = true`
- `sionna_rzf_callable = true`
- `converted_to_precoder_output = true`
- `native_receiver_success_project = true`
- `native_receiver_success_sionna = true`
- `relationship_status = close_but_different`
- `semantic_compatibility_passed = true`
- `strict_equivalence_claim_allowed = false`

Current measured differences on the validated shared realization:

- `max_abs_diff_f_f_if_comparable = 0.061414435505867004`
- `abs_diff_sum_rate = 0.09229850769042969`
- `abs_diff_symbol_mse = 0.0006549134850502014`
- `abs_diff_sinr_db = 0.07219910621643066`

Interpretation:

- the native method is compatible enough to keep as an optional adapter-backed method
- the project bridge and native bridge are still not strictly numerically equivalent
- the correct wording is `close but different`, not `strict equivalent`

## Quick SNR / seed sweep

The quick sweep checks whether the semantic gap behaves consistently across:

- `seeds = 1,2,3`
- `snr = 0,5,10,15,20 dB`

Current quick result:

- `RZFPrecoder` is callable on all evaluated rows
- conversion to `PrecoderOutput` succeeds on all evaluated rows
- native receiver success is true on all evaluated rows
- semantic compatibility passes on all evaluated rows
- strict numerical equivalence is still false on all evaluated rows
- all evaluated rows remain `close_but_different`

This means:

- `sionna_rzf_precoder` can now be included as an optional native method
- the adapter appears stable enough for release-hardening-level integration
- it still does not support a strict equivalence claim against `project_rzf`

## Optional native method status

Current method-level interpretation:

- `method = sionna_rzf_precoder`
- `source = sionna_rzf_precoder`
- `sionna_native_precoder = true`
- `project_side_precoder = false`
- `full_native_only = false`

This status is valid for the adapter-produced native output object itself.

It is **not** the same as claiming:

- project-side precoder replacement is complete
- `project_rzf` and `sionna_rzf_precoder` are strictly equivalent
- the benchmark is now full native-only

## v0.9.0 Candidate Status

Current `v0.9.0` candidate interpretation:

- `sionna_rzf_precoder` is now a release-hardened optional native method bridge candidate
- the native output can be converted to `PrecoderOutput`
- the converted output can enter the current native receiver path
- the same-realization relationship remains `close_but_different`
- strict equivalence to `project_rzf` is still not supported

Compact native precoder table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| `sionna_rzf_available` | `true` | native `RZFPrecoder` exists on the current Sionna install |
| `sionna_rzf_callable` | `true` | native call path is operational |
| `converted_to_precoder_output` | `true` | adapter bridge maps native output into project schema |
| `native_receiver_success` | `true` | converted native output traverses the current receiver path |
| `relationship_status` | `close_but_different` | semantic alignment is strong enough for optional integration |
| `strict_equivalence_claim_allowed` | `false` | no strict project_rzf equivalence claim |
| `full_native_only` | `false` | still not a full native-only benchmark |

## Native Precoder Contract

The bridge is now treated as a contract-checked optional method rather than a loose probe only.

Current contract points:

- native Sionna input:
  - `x = [B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]`
  - `h = [B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]`
- project interfaces:
  - `H_f = (B,Nsc,K,Nt)`
  - `F_f = (B,Nsc,Nt,K)`
- alignment rule:
  - adapter bridge required
  - not a direct drop-in replacement
- semantic rule:
  - `relationship_status = close_but_different`
  - `strict_equivalence_claim_allowed = false`
- identity rule:
  - `sionna_native_precoder = true` only for the adapter-generated Sionna output object
  - `project_side_precoder = false` for that object

## Skip and Fallback Policy

Current hardened rules are:

- if Sionna is missing:
  - skip `sionna_rzf_precoder`
  - do not fail the whole demo
  - record `reason = sionna_not_installed`
- if `RZFPrecoder` is unavailable or not callable:
  - skip
  - do not alias `project_rzf` as `sionna_rzf_precoder`
- if adapter mapping/conversion fails:
  - skip
  - record `adapter_failure_reason`
- if native receiver fails after a successful conversion:
  - keep the probe result
  - record `native_receiver_success = false`
  - do not claim receiver-compatible success

## v1.0.0-rc1 Candidate View

Current RC framing is:

- interface-first bridge candidate
- optional Sionna dependency only
- contract-hardened optional native method
- not a mainline native replacement
- not full native-only

End-to-end interface text chain:

- `Sionna OFDMChannel -> ExtractedCSI -> PrecoderOutput -> native receiver path`

Current stable release view:

- `v1.0.0` is the stable interface-first release
- stable finalization reuses the RC minimal reproduction plus consistency/provenance/smoke audits
- the stable wording still preserves:
  - optional dependency only
  - not a mainline native replacement
  - not strict `project_rzf` equivalence
  - not full native-only

## Current boundary

The supported boundary remains:

- optional dependency only
- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- no stable learned `> WMMSE-iter5` claim
