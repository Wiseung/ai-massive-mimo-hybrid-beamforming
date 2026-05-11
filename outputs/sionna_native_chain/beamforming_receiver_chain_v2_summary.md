# Sionna Native OFDM Beamforming Chain

- Demo status: `ok`
- receiver_mode: `auto`
- native_receiver_attempted: `True`
- native_receiver_success: `True`
- Used Sionna ResourceGrid: `True`
- Used Sionna channel: `True`
- Used Sionna estimator: `True`
- Used Sionna equalizer: `True`
- Used Sionna demapper: `True`
- Fallback used: `True`
- shape_trace_path: `outputs/sionna_native_chain/beamforming_receiver_shape_trace_runtime.json`

| Method | Native OK | BER | Symbol MSE | Effective SINR dB | Approx Sum Rate | Fallback | Stage | Reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| no_precoding | True | 0.3486328125 | 16.060356 | -12.057552 | 0.348576 | False |  |  |
| project_rzf | True | 0.44775390625 | 31.768276 | -15.019938 | 0.178852 | False |  |  |
| project_wmmse_iter_2 | True | 0.455078125 | 32.501076 | -15.118979 | 0.174879 | False |  |  |
| project_wmmse_iter_5 | True | 0.45458984375 | 14.543175 | -11.626593 | 0.383756 | False |  |  |

## Notes
- Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable.
- Pilot-aware native beamformed receiver mode uses num_tx=1, num_streams_per_tx=K, rx_tx_association=ones(K,1), and at least one non-pilot OFDM symbol.
- receiver-mode=proxy keeps project-side proxy metrics only.
- receiver-mode=native requires a real Sionna receiver path and records exact failure stage/reason if it fails.
- receiver-mode=auto attempts the native receiver first and falls back to proxy metrics if needed.
