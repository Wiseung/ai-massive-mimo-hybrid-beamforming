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

## Learned Beamformer Insertion Recommendation

Yes, this is now the recommended insertion point for the next learned-beamformer phase:

- keep the beamformer in frequency domain
- operate per subcarrier on `H_f = (B, Nsc, K, Nt)`
- output `F_f = (B, Nsc, Nt, K)`
- feed the result into the Sionna-native OFDM chain where possible

This recommendation is about architecture and integration cleanliness, not about claiming the chain is already a production-ready full Sionna e2e path.

## Limitations

- optional Sionna dependency only: `sionna-no-rt`
- synthetic/channel-level only
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- not a production e2e system
