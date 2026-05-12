# CSI-backed Beamforming Summary

- csi_interface_used: `True`
- csi_source: `sionna_ofdm_channel`
- project_h_f_assisted: `False`
- extracted_h_f_used: `True`
- full_native_only: `False`
- native_receiver_success: `True`
- precoder_interface_used: `False`

| Method | Precoder Input | Native OK | Teacher Inference | Sum Rate | Fallback | Reason |
| --- | --- | --- | --- | ---: | --- | --- |
| project_rzf | raw_f_f | True | False | 18.545010 | False |  |
| project_wmmse_iter_5 | raw_f_f | True | False | 18.354210 | False |  |
| learned_residual_rzf | raw_f_f | True | False | 18.546923 | False |  |
| learned_residual_wmmse_distill | raw_f_f | True | False | 18.546631 | False |  |
