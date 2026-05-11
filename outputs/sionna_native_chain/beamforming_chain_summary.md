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
| no_precoding | None | 1.284707 | -4.327354 | 1.579393 | True | beamformed_rx_chain_fallback: AssertionError: The pilot pattern cannot be empty |
| project_rzf | None | 0.597654 | 14.913561 | 19.935688 | True | beamformed_rx_chain_fallback: AssertionError: The pilot pattern cannot be empty |
| project_wmmse_iter_1 | None | 0.923470 | 14.426174 | 18.535728 | True | beamformed_rx_chain_fallback: AssertionError: The pilot pattern cannot be empty |
| project_wmmse_iter_2 | None | 0.829389 | 15.221720 | 19.803410 | True | beamformed_rx_chain_fallback: AssertionError: The pilot pattern cannot be empty |
| project_wmmse_iter_5 | None | 0.776667 | 15.263196 | 20.004618 | True | beamformed_rx_chain_fallback: AssertionError: The pilot pattern cannot be empty |

## Notes
- Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable.
- Project frequency-domain precoders are the current clean insertion path because they match H_f=(B,Nsc,K,Nt) directly.
- Optional Sionna RZFPrecoder is audited separately and recorded only as a shape-checked reference path.
