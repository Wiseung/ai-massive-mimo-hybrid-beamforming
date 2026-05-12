# Project vs Sionna Native Precoder Comparison

1. Sionna RZFPrecoder callable: `True`
2. converted to PrecoderOutput: `True`
3. enters native receiver path: `True`
4. shape/power close: `shape_compatible=True`, `power_norm_project=1.0`, `power_norm_sionna=1.0`
5. sionna_native_precoder=true allowed: `True`
6. full native-only benchmark: `False`
7. recommended next step: `adapter_bridge_then_optional_native_receiver_probe`

Current interpretation:
- this phase establishes API/shape compatibility mapping, not a project-side precoder replacement claim
- if converted_to_precoder_output=true, the adapter bridge is working for the current minimal probe
- the benchmark boundary still remains not full native-only
