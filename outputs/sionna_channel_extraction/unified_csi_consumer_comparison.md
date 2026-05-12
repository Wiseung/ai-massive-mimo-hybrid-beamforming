# Unified CSI Consumer Comparison

- comparison_type: `cross_run_comparison`
- same_seed_used: `True`
- same_csi_tensor_signature: `False`
- strict_equivalence_claim_allowed: `False`

1. unified demo matches existing CSI-backed path within tolerance: `False`
2. all key consumers accept ExtractedCSI: `True`
3. additional fallback introduced: `False`
4. provenance clarity improved: `True`.
5. full native-only benchmark completed: `False`.

- baseline_ranking: `['learned_residual_rzf', 'learned_residual_wmmse_distill', 'project_rzf', 'project_wmmse_iter_5']`
- unified_ranking: `['learned_residual_rzf', 'learned_residual_wmmse_distill', 'project_rzf', 'project_wmmse_iter_5']`

This artifact compares separate reruns. It should not be interpreted as a strict same-batch equivalence test.
