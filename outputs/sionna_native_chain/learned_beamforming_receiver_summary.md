# Sionna Native Learned Beamforming Chain

- Demo status: `ok`
- native_receiver_attempted: `True`
- methods_successful_under_native_receiver: `['no_precoding', 'project_rzf', 'project_wmmse_iter_2', 'project_wmmse_iter_5', 'learned_residual_rzf', 'learned_residual_wmmse_distill']`
- methods_skipped_missing_checkpoint: `[]`

| Method | Type | Native OK | Teacher Inference | BER | MSE | SINR dB | Sum Rate | Fallback | Reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| no_precoding | reference | True | False | 0.32275390625 | 153.004074 | -21.847031 | 0.037594 | False |  |
| project_rzf | analytic | True | False | 0.0 | 0.043652 | 13.599933 | 18.317764 | False |  |
| project_wmmse_iter_2 | analytic | True | False | 0.0 | 0.045776 | 13.393624 | 18.055357 | False |  |
| project_wmmse_iter_5 | analytic | True | False | 0.0 | 0.043248 | 13.640367 | 18.369255 | False |  |
| learned_residual_rzf | learned | True | False | 0.0 | 0.041295 | 13.841002 | 18.625042 | False |  |
| learned_residual_wmmse_distill | learned | True | False | 0.0 | 0.042724 | 13.693235 | 18.436609 | False |  |

## Notes
- Receiver chain is a real Sionna-native path, but precoder/H_f remains project-assisted.
- This is still a synthetic/channel-level benchmark, not a full native-only or production e2e system.
