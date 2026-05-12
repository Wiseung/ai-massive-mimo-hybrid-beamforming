# PrecoderOutput Comparison Mismatch Audit

- comparison_type: `cross_run_comparison`
- comparison_independent_runs: `True`
- same_seed_used: `True`
- same_csi_object_used: `False`
- same_raw_f_f_used: `False`
- same_receiver_config_used: `False`
- interface_bug_evidence: `False`

Root cause: previous raw-F_f-vs-PrecoderOutput mismatch was a cross-run comparison without a shared CSI/Precoder realization, not direct PrecoderOutput interface bug evidence.
