# CSI-backed Beamforming Summary

- csi_interface_used: `True`
- csi_source: `sionna_ofdm_channel`
- project_h_f_assisted: `False`
- extracted_h_f_used: `True`
- full_native_only: `False`
- native_receiver_success: `True`

| Method | Native OK | Teacher Inference | Sum Rate | Fallback | Reason |
| --- | --- | --- | ---: | --- | --- |
| project_rzf | True | False | 18.564106 | False |  |
| project_wmmse_iter_5 | True | False | 17.828186 | False |  |
| learned_residual_rzf | True | False | 18.394684 | False |  |
| learned_residual_wmmse_distill | True | False | 18.430624 | False |  |
