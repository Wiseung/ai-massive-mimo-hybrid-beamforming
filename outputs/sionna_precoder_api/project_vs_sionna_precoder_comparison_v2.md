# Project vs Sionna Native Precoder Comparison v2

1. Sionna RZFPrecoder callable: `True`
2. converted to PrecoderOutput: `True`
3. enters native receiver path: `same_realization=True`, `unified_demo=True`
4. semantic compatibility: `True` with relationship `close_but_different`
5. sionna_native_precoder=true allowed: `True`
6. project_rzf strict equivalence allowed: `False`
7. full native-only benchmark: `False`
8. recommended next step: `release_hardening`

Current interpretation:
- same-realization validation is the only valid place to judge strict numerical equivalence
- quick alignment sweep is used to judge stability of the semantic gap across seeds and SNRs
- if semantic_compatibility_passed=true, `sionna_rzf_precoder` can be kept as an optional native method behind explicit Sionna availability checks
- this still does not justify a full native-only benchmark claim
