# Sionna Native OFDM Beamforming Chain

- Demo status: `ok`
- Used Sionna ResourceGrid: `True`
- Used Sionna channel: `False`
- Used Sionna estimator: `False`
- Used Sionna equalizer: `False`
- Used Sionna demapper: `False`
- Fallback used: `True`

| Method | BER | Symbol MSE | Effective SINR dB | Approx Sum Rate | Fallback | Reason |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| no_precoding | None | 1.257635 | -4.081345 | 1.675143 | True | RuntimeError: shape '[16, 1, 1, 0]' is invalid for input of size 2048 |
| project_rzf | None | 0.589396 | 14.879130 | 19.877598 | True | RuntimeError: shape '[16, 1, 1, 0]' is invalid for input of size 2048 |
| project_wmmse_iter_2 | None | 0.826376 | 15.210633 | 19.765434 | True | RuntimeError: shape '[16, 1, 1, 0]' is invalid for input of size 2048 |
| project_wmmse_iter_5 | None | 0.775069 | 15.245140 | 19.953991 | True | RuntimeError: shape '[16, 1, 1, 0]' is invalid for input of size 2048 |

## Notes
- Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable.
- Pilot-enabled ResourceGrid is required when --enable-receiver-chain is used.
- Project frequency-domain precoders are the current clean insertion path because they match H_f=(B,Nsc,K,Nt) directly.
- Optional Sionna RZFPrecoder is audited separately and recorded only as a shape-checked reference path.
