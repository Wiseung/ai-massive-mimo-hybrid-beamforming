# Unified CSI + PrecoderOutput Demo

- csi_interface_used: `True`
- same_csi_object_used_for_all_methods: `True`
- precoder_interface_used: `True`
- all_precoders_emit_precoder_output: `True`
- all_receiver_consumers_accept_precoder_output: `True`
- native_receiver_success: `True`
- no_new_fallback_introduced: `True`
- sionna_rzf_available: `True`
- sionna_rzf_callable: `True`
- sionna_rzf_evaluated: `True`
- sionna_rzf_skipped_reason: ``

- failed_methods: `[]`
- methods_evaluated: `['project_rzf', 'project_wmmse_iter_5', 'learned_residual_rzf', 'learned_residual_wmmse_distill', 'sionna_rzf_precoder']`

| Method | Precoder Input | Native OK | Teacher Inference | Sum Rate | Fallback | Reason |
| --- | --- | --- | --- | ---: | --- | --- |
| project_rzf | PrecoderOutput | True | False | 18.363405 | False |  |
| project_wmmse_iter_5 | PrecoderOutput | True | False | 17.789713 | False |  |
| learned_residual_rzf | PrecoderOutput | True | False | 18.362823 | False |  |
| learned_residual_wmmse_distill | PrecoderOutput | True | False | 18.364584 | False |  |
| sionna_rzf_precoder | PrecoderOutput | True | False | 18.247169 | False |  |
