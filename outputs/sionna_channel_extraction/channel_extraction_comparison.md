# Sionna Channel Extraction Comparison

1. success extracting H_f from Sionna channel tensor: `True`
2. extracted H_f compatible with project precoder interface: `True`
3. native-channel-assisted chain success: `True`
4. project_h_f_assisted limitation reduced: `True`
5. learned_residual_rzf can use extracted H_f: `True`
6. full native-only benchmark achieved: `False`
7. next step: keep reducing project-assisted assumptions on the channel/precoder side while preserving the native Sionna receiver path.

## Notes
- Current interpretation remains synthetic and optional-Sionna only.
- Even with extracted H_f, this should not yet be described as a full native-only benchmark unless the full channel/precoder/receiver stack is consistently native.
