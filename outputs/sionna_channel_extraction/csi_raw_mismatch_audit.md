# CSI Raw-vs-CSI Mismatch Audit

- comparison_independent_runs: `True`
- same_seed_used: `True`
- same_channel_tensor_shape_metadata: `True`
- same_selected_ofdm_symbol: `True`
- same_effective_subcarrier_indices: `True`
- same_receiver_mode: `True`
- csi_interface_bug_evidence: `False`

Root cause: prior raw-vs-CSI comparison was a cross-run comparison, not a strict same-batch equivalence test.
