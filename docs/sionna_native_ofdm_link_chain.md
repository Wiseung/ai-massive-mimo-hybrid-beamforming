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

## Limitations

- optional Sionna dependency only: `sionna-no-rt`
- synthetic/channel-level only
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- not a production e2e system
