# Sionna Native Learned Beamforming Chain

- Demo status: `ok`
- native_receiver_attempted: `True`
- methods_successful_under_native_receiver: `['no_precoding', 'project_rzf', 'project_wmmse_iter_2', 'project_wmmse_iter_5', 'learned_residual_rzf', 'learned_residual_wmmse_distill']`
- methods_skipped_missing_checkpoint: `[]`

| Method | Type | Native OK | Teacher Inference | BER | MSE | SINR dB | Sum Rate | Fallback | Reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| no_precoding | reference | True | False | 0.33447265625 | 13.186584 | -11.201323 | 0.421825 | False |  |
| project_rzf | analytic | True | False | 0.0 | 0.040820 | 13.891314 | 18.689259 | False |  |
| project_wmmse_iter_2 | analytic | True | False | 0.0 | 0.048855 | 13.110870 | 17.696611 | False |  |
| project_wmmse_iter_5 | analytic | True | False | 0.0 | 0.043787 | 13.586547 | 18.300722 | False |  |
| learned_residual_rzf | learned | True | False | 0.0 | 0.044183 | 13.547403 | 18.250900 | False |  |
| learned_residual_wmmse_distill | learned | True | False | 0.0 | 0.043155 | 13.649725 | 18.381174 | False |  |

## Notes
- Receiver chain is a real Sionna-native path, but precoder/H_f remains project-assisted.
- This is still a synthetic/channel-level benchmark, not a full native-only or production e2e system.
