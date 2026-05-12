# Unified CSI Consumers Demo

- csi_object_created: `True`
- same_csi_object_used_for_all_methods: `True`
- all_consumers_accept_csi: `True`
- native_receiver_success: `True`
- no_new_fallback_introduced: `True`

- failed_consumers: `[]`
- methods_evaluated: `['project_rzf', 'project_wmmse_iter_5', 'learned_residual_rzf', 'learned_residual_wmmse_distill']`

| Method | Type | Input | Native OK | Teacher Inference | Sum Rate | Fallback | Reason |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| project_rzf | analytic | ExtractedCSI | True | False | 18.727100 | False |  |
| project_wmmse_iter_5 | analytic | ExtractedCSI | True | False | 18.498144 | False |  |
| learned_residual_rzf | learned | ExtractedCSI | True | False | 18.730652 | False |  |
| learned_residual_wmmse_distill | learned | ExtractedCSI | True | False | 18.729382 | False |  |
