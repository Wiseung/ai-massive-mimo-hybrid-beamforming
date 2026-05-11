# Sionna Native OFDM Link Chain

This phase extends the optional Sionna work toward a more Sionna-native OFDM link chain. The existing `v0.3.0` learned training pipeline remains a torch-heavy OFDM training stack with explicit Sionna component usage where practical. The goal here is narrower: verify a cleaner Sionna PHY/OFDM chain for mapping, channel application, estimation, equalization, and demapping without claiming a production end-to-end system.

## Scope

- Sionna-native OFDM component audit
- Sionna-first OFDM baseline chain
- frequency-domain, per-subcarrier processing first
- synthetic/channel-level only
- no Sionna RT
- no ray tracing
- no 5G NR full stack

## Why This Phase Exists

The current learned training path relies heavily on PyTorch-side signal generation, channel simulation glue, and custom metrics. That remains useful for experimentation, but it does not answer a separate integration question: how much of the OFDM link can already be executed through real Sionna 2.x PHY/OFDM blocks on this repository setup.

This phase therefore focuses on a Sionna-native baseline chain before any new beamforming model integration. It is still not a production e2e system and should not be confused with a full Sionna NR stack.

## Recommended Beamforming Insertion Point

Current recommendation:

- use frequency-domain, per-subcarrier beamforming first
- insert the beamformer before OFDM channel application
- keep the representation aligned with `ResourceGrid` / effective subcarriers

This is preferred over trying to insert beamforming after `ResourceGrid` mapping in a way that blurs stream structure, and it avoids prematurely attempting a full 5G NR transmit/receive stack.

## Native Chain Components

The current target component path is:

- `BinarySource`
- `Mapper`
- `ResourceGrid`
- `ResourceGridMapper`
- `OFDMChannel` or `ApplyOFDMChannel`
- `LSChannelEstimator`
- `LMMSEEqualizer`
- `Demapper`

Auxiliary components include:

- `ResourceGridDemapper`
- `RemoveNulledSubcarriers`
- `StreamManagement`
- `RayleighBlockFading`

## Integration Notes

- `OFDMChannel` and `ApplyOFDMChannel` must be resolved from the real Sionna 2.0.1 install, not assumed from older `sionna.phy.ofdm` paths.
- If estimator/equalizer/demapper shapes are unstable for the current API surface, the scripts must record explicit fallback flags instead of pretending that the chain is fully native.
- The next beamforming phase should target frequency-domain per-subcarrier precoding, not a full NR stack migration.

## Precoding Audit Result

- `RZFPrecoder` is available in `sionna.phy.ofdm` and its constructor/call path is real on this install.
- `RZFPrecoder` expects a Sionna resource-grid tensor plus a higher-rank channel tensor layout, not the repository's simpler `H_f = (B, Nsc, K, Nt)` shape.
- `PrecodedChannel` is available as an effective-channel helper, but it is not the clean primary insertion point for the repository's project-side per-subcarrier precoders.
- Current recommendation:
  use project frequency-domain precoder insertion first, and keep `RZFPrecoder` as an optional shape-checked reference path.

## Beamforming Chain Status

Current demo compares:

- `no_precoding`
- `project_rzf`
- `project_wmmse_iter_1`
- `project_wmmse_iter_2`
- `project_wmmse_iter_5`
- optional `sionna_rzf_precoder` audit/reference path

Observed status on this branch:

- real Sionna `ResourceGrid` is still used
- the beamforming demo currently uses a synthetic Rayleigh `H_f` fallback instead of extracting a compatible beamformed channel tensor from Sionna
- the multi-user beamformed RX path does not yet reuse the Sionna estimator/equalizer/demapper chain cleanly because the current setup trips over empty-pilot assumptions
- therefore the current beamforming demo reports `symbol_mse`, `effective_sinr_db`, `approximate_sum_rate`, `power_norm`, and `power_violation` as the reliable metrics

Current method comparison from the demo:

- `project_rzf` strongly improves over `no_precoding`
- `project_wmmse_iter_5` slightly improves `approximate_sum_rate` over `project_rzf`
- this does not imply any learned model exceeds `WMMSE-iter5`; it only reflects project analytic precoders inside the current native-chain insertion experiment

## Pilot Pattern Audit

Current pilot-pattern conclusion:

- `pilot_pattern=None` creates `EmptyPilotPattern`
- `LSChannelEstimator` fails immediately on `EmptyPilotPattern`
- `pilot_pattern="kronecker"` with `pilot_ofdm_symbol_indices=[0]` is the current minimal working configuration

This means the receiver-chain integration problem has two layers:

1. pilot-free ResourceGrid is invalid for the estimator path
2. after fixing pilots, the beamformed multi-user shape still needs an explicit bridge

## Minimal Receiver Chain Demo

The minimal success path now exists without beamforming:

- `ResourceGrid`
- `OFDMChannel`
- `LSChannelEstimator`
- `LMMSEEqualizer`
- `Demapper`

This confirms that the current install can run a real pilot-based Sionna receiver chain when the stream count and grid configuration stay in the simple supported regime.

## Beamformed Receiver Chain Status

Current status for `--enable-receiver-chain`:

- the chain now uses a pilot-enabled `ResourceGrid`
- the shape-trace audit localized the earlier `shape '[16,1,1,0]'` failure to a pilot-only grid:
  `num_ofdm_symbols=1` with `pilot_ofdm_symbol_indices=[0]` forces `num_data_symbols=0`
- the StreamManagement audit shows the beamformed downlink should be modeled as
  `num_tx=1`, `num_streams_per_tx=K`, `rx_tx_association=ones(K,1)`
- with that bridge plus `ApplyOFDMChannel`, the current native beamformed receiver retry succeeds for:
  `no_precoding`, `project_rzf`, `project_wmmse_iter_2`, `project_wmmse_iter_5`

This is now a real Sionna-native beamformed receiver path for the current synthetic link-level experiment. It is still not a production e2e system and it still does not change any release claim about RT, ray tracing, or a full NR stack.

## Shape Trace And StreamManagement Audit

Current conclusions:

- minimal success path shapes are stable:
  `y = [B, 1, 1, 4, 16]`,
  `h_hat = [B, 1, 1, 1, 1, 4, 13]`,
  `err_var = [B, 1, 1, 1, 1, 4, 13]`,
  `x_hat = [B, 1, 1, 39]`,
  `no_eff = [B, 1, 1, 39]`
- the old beamformed failure did not come from an arbitrary equalizer bug:
  it came from `num_data_symbols=0`, which then propagated into `x_hat/no_eff = [16, 1, 1, 0]`
- `StreamManagement(np.ones((K,1)), num_streams_per_tx=1)` is not the correct downlink semantic mapping for the project beamformed chain
- the current recommended bridge is:
  build a pilot-aware `ResourceGrid` with at least one data OFDM symbol,
  set `num_tx=1`,
  set `num_streams_per_tx=K`,
  set `rx_tx_association=np.ones((K,1))`,
  map project stream symbols into the data REs only,
  and apply project precoders onto the antenna-domain grid before `ApplyOFDMChannel`

This keeps the project-side `H_f=(B,Nsc,K,Nt)` and `F_f=(B,Nsc,Nt,K)` interface while making the Sionna-native receiver chain interpretable.

## Current Native Receiver Outcome

The current `receiver-mode=auto` run reports:

- `native_receiver_attempted = true`
- `native_receiver_success = true`
- native-success methods:
  `no_precoding`, `project_rzf`, `project_wmmse_iter_2`, `project_wmmse_iter_5`
- no methods required fallback-only handling in the current validated run

This means the repository now has both:

- a minimal non-beamformed Sionna receiver success path
- a beamformed Sionna receiver success path through the new shape bridge

The bridge remains experimental and synthetic-only, but it is now strong enough to justify attaching the next learned frequency-domain model at this insertion point.

## Learned Beamformer Insertion Recommendation

Yes, this is now the recommended insertion point for the next learned-beamformer phase:

- keep the beamformer in frequency domain
- operate per subcarrier on `H_f = (B, Nsc, K, Nt)`
- output `F_f = (B, Nsc, Nt, K)`
- feed the result into the Sionna-native OFDM chain where possible

This recommendation is about architecture and integration cleanliness, not about claiming the chain is already a production-ready full Sionna e2e path.

The recommended next model family remains `residual_rzf`, because it already matches the project-side frequency-domain precoder interface and does not require any claim beyond the current optional experimental scope.

## Limitations

- optional Sionna dependency only: `sionna-no-rt`
- synthetic/channel-level only
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- not a production e2e system
