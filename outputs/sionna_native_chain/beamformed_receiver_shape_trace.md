# Beamformed Receiver Shape Trace

## Summary

1. minimal success path y/h_hat/err_var/x_hat/no_eff shapes are: `[32, 1, 1, 4, 16]` / `[32, 1, 1, 1, 1, 4, 13]` / `[32, 1, 1, 1, 1, 4, 13]` / `[32, 1, 1, 39]` / `[32, 1, 1, 39]`
2. previous beamformed path failed at stage `resource_grid_construction`
3. shape `[16,1,1,0]` came from `num_data_symbols=0 because num_ofdm_symbols=1 and pilot_ofdm_symbol_indices=[0] leave no data OFDM symbol`
4. the 0 dimension is a `num_data_symbols` problem caused by a pilot-only grid, not a random CUDA issue
5. recommended fix is `num_tx=1, num_streams_per_tx=K`, `rx_tx_association=ones(K,1)`, and at least one non-pilot OFDM symbol

## Native Retry
- success: `True`
- BER: `0.0`
- symbol_mse: `0.042959995567798615`
