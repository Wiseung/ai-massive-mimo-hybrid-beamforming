# StreamManagement Audit

## Summary

1. minimal receiver demo succeeds because: The minimal chain uses num_tx=1, num_streams_per_tx=1, rx_tx_association=[[1]], and a pilot-aware grid with positive num_data_symbols.
2. beamformed chain should map K users as: `Model the downlink beamformed chain as num_tx=1 and num_streams_per_tx=K, not as K receivers each fed by a single-stream transmitter.`
3. recommended rx_tx_association: `[[1], [1], [1], [1]]`
4. zero dimension came directly from StreamManagement: `False`

| Case | num_rx | num_tx | num_streams_per_tx | desired_ind | undesired_ind |
| --- | ---: | ---: | ---: | --- | --- |
| minimal_success | 1 | 1 | 1 | `[0]` | `[]` |
| previous_beamformed_failing_sm | 4 | 1 | 1 | `[]` | `[0, 1, 2, 3]` |
| recommended_beamformed_native_sm | 4 | 1 | 4 | `[0, 5, 10, 15]` | `[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]` |
