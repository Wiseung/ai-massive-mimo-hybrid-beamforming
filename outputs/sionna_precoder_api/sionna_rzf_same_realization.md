# Sionna RZF Same-realization Validation

- comparison_type: `same_realization_comparison`
- same_realization_comparison: `True`
- same_csi_object_used: `True`
- same_symbols_used: `True`
- same_receiver_config_used: `True`
- same_noise_config_used: `True`
- converted_to_precoder_output: `True`
- native_receiver_success_project: `True`
- native_receiver_success_sionna: `True`
- relationship_status: `close_but_different`
- semantic_compatibility_passed: `True`
- strict_equivalence_claim_allowed: `False`
- suggested_for_optional_method_list: `True`

- max_abs_diff_f_f_if_comparable: `0.05982483550906181`
- abs_diff_sum_rate: `0.0704193115234375`
- abs_diff_symbol_mse: `0.0005603842437267303`
- abs_diff_sinr_db: `0.055327415466308594`
- difference_primary_axis: `sum_rate`

- conclusion: `close_but_different_under_shared_realization`
- notes: `['This artifact is a same-realization comparison with one shared ExtractedCSI object, one shared symbol batch, and one shared native receiver configuration.', 'strict_equivalence_claim_allowed only becomes true if F_f, sum-rate, symbol MSE, and SINR all match within tolerance.']`
