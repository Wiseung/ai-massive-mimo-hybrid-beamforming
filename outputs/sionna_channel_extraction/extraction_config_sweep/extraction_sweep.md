# Sionna Channel Extraction Config Sweep

- quick: `True`
- resource_grid_meta: `{'fallback_used': False, 'fallback_reason': '', 'num_users': 4, 'num_tx': 1, 'num_streams_per_tx': 4, 'fft_size': 19, 'num_data_symbols': 32, 'num_pilot_symbols': 16, 'effective_subcarrier_ind': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17], 'rx_tx_association': [[1], [1], [1], [1]]}`

## Key answers
1. selected OFDM symbol sensitivity (mean fro_norm): `{'all_data_average': 4.632591088612874, 'first_data': 4.462871313095093, 'last_data': 4.538872400919597}`
2. effective subcarrier sensitivity (mean fro_norm): `{'all_effective': 4.572952111562093, 'center_16': 4.571678161621094, 'center_8': 4.489704529444377}`
3. extracted H_f norm/rank stable across successful rows: `True`
4. recommended default extraction config: `selected_ofdm_symbol=first_data`, `effective_subcarriers=all_effective`, `normalize_channel=false`.

This sweep reduces ambiguity around axis/config choices, but it does not justify a full native-only benchmark claim.
