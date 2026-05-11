# Sionna Native OFDM Baseline Chain

- Demo status: `ok`
- Sionna import ok: `True`
- Sionna version: `2.0.1`
- Used Sionna native components: `True`
- Fallback used: `False`

- BER if available: `0.03565705195069313`
- Symbol MSE: `0.40306782722473145`
- Empirical SNR dB: `3.946218192577362`

## Components
- `BinarySource`
- `Mapper`
- `ResourceGrid`
- `ResourceGridMapper`
- `ResourceGridDemapper`
- `RemoveNulledSubcarriers`
- `RayleighBlockFading`
- `OFDMChannel`
- `LSChannelEstimator`
- `LMMSEEqualizer`
- `Demapper`

## Notes
- Used real Sionna OFDMChannel with RayleighBlockFading.
- Used real Sionna LSChannelEstimator and LMMSEEqualizer. Effective noise variance mean=0.387411.
- Used real Sionna Demapper for hard-bit BER.
- Recommended beamforming insertion point for the next phase is frequency-domain per-subcarrier precoding before OFDMChannel.
- This remains a synthetic channel-level smoke chain, not a full 5G NR or production e2e stack.
