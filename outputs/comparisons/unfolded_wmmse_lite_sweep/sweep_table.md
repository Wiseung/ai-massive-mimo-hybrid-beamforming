# Unfolded WMMSE-lite Sweep

| tag | source | init | layers | distill | delta | mean_se | gap_to_wmmse | gap_to_wmmse_iter_5 | latency_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wmmse_iter_5_L3_dw0p1_delta0p001 | quick | wmmse_iter_5 | 3 | 0.1 | 0.001 | 5.816346 | -0.4282% | +0.0055% | 189.214 |
| existing_wmmse_iter_2_L5_dw0p1_delta0p01 | existing | wmmse_iter_2 | 5 | 0.1 | 0.01 | 5.772821 | -1.5051% | -1.0774% | 80.740 |
| wmmse_iter_2_L3_dw0p1_delta0p001 | quick | wmmse_iter_2 | 3 | 0.1 | 0.001 | 5.771418 | -1.5144% | -1.0869% | 77.915 |
| wmmse_iter_2_L3_dw0p0_delta0p001 | quick | wmmse_iter_2 | 3 | 0.0 | 0.001 | 5.771418 | -1.5144% | -1.0869% | 78.347 |
| wmmse_iter_1_L3_dw0p1_delta0p001 | quick | wmmse_iter_1 | 3 | 0.1 | 0.001 | 5.700295 | -3.2000% | -2.7821% | 41.332 |
| rzf_L3_dw0p1_delta0p001 | quick | rzf | 3 | 0.1 | 0.001 | 5.582462 | -9.4057% | -9.0132% | 2.697 |
