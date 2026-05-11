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

| Method | Native OK | BER | Symbol MSE | Effective SINR dB | Approx Sum Rate | Fallback | Stage | Reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| no_precoding | True | 0.35302734375 | 15.789111 | -11.983578 | 0.354384 | False |  |  |
| project_rzf | True | 0.4541015625 | 19.762955 | -12.958519 | 0.284852 | False |  |  |
| project_wmmse_iter_2 | True | 0.45947265625 | 17.899153 | -12.528325 | 0.313721 | False |  |  |
| project_wmmse_iter_5 | True | 0.4599609375 | 51.372116 | -17.107276 | 0.111254 | False |  |  |

## Notes
- Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable.
- Pilot-aware native beamformed receiver mode uses num_tx=1, num_streams_per_tx=K, rx_tx_association=ones(K,1), and at least one non-pilot OFDM symbol.
- receiver-mode=proxy keeps project-side proxy metrics only.
- receiver-mode=native requires a real Sionna receiver path and records exact failure stage/reason if it fails.
- receiver-mode=auto attempts the native receiver first and falls back to proxy metrics if needed.
