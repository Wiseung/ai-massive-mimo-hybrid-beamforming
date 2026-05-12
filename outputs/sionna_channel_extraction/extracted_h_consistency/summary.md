# Extracted H_f Consistency Benchmark

- quick: `True`
- batch_size: `4`
- seeds: `[1, 2, 3]`
- snrs_db: `[0.0, 5.0, 10.0, 15.0, 20.0]`
- extraction_success_all_available_rows: `True`
- native_receiver_success_all_available_rows: `True`

## Key answers
1. extracted H_f stable usable: `True`
2. proxy/native method ranking exact-agreement rate: `0.226667`
3. learned_residual_rzf vs project_rzf mean gap: `-1.681560%`; positive-fraction across seed/SNR: `0.466667`
4. learned_residual_rzf vs project_wmmse_iter_5 mean gap: `4.735929%`; positive-fraction across seed/SNR: `0.866667`
5. interpretation label should remain: `native-channel-assisted`, not `full native-only benchmark`.
