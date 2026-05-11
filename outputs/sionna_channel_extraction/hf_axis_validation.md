# Sionna Extracted H_f Axis Validation

- validation_status: `ok`
- sionna_channel_tensor_shape: `[8, 4, 1, 1, 16, 2, 19]`
- extracted_h_f_shape: `[8, 16, 4, 16]`
- selected_data_symbol_indices: `[1]`
- selected_effective_subcarrier_indices: `[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]`
- axis_spot_check_passed: `True`
- spot_check_max_abs_diff: `0.0`
- selected_ofdm_symbol_is_data_bearing: `True`
- effective_subcarrier_count_matches_nsc: `True`
- hidden_squeeze_transpose_risk: `low_for_current_num_tx_equals_1_and_rx_ant_equals_1_bridge_but_still_explicit_if_future_multi_tx_or_multi_rx_ant_paths_are_added`

## Matrix stats
- available: `True`
- rank_mean: `4.0`
- rank_min: `4`
- rank_max: `4`
- condition_number_mean: `2.022429943084717`
- condition_number_max: `2.3918986320495605`
- fro_norm_mean: `8.04751205444336`
- fro_norm_std: `0.45754683017730713`

## Notes
- Current converter assumes Sionna axes [batch, rx, rx_ant, tx, tx_ant, ofdm_symbol, fft_bin] and project axes [batch, subcarrier, user, bs_ant].
- Selected OFDM symbol indices are data-bearing indices [1]; available data indices are [1].
- Selected effective FFT-bin indices are [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17].
- Spot-check compares every extracted H_f[:, :, k, n] slice against the corresponding raw Sionna h[:, k, 0, 0, n, ofdm_symbol, fft_bin] slice.
