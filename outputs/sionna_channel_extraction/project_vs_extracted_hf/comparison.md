# Project-H_f vs Extracted-H_f Comparison

1. extracted-H_f path changes ranking relative to project-assisted single run: `True`
2. learned_residual_rzf close to analytic baseline on both paths: `True`
3. WMMSE-iter5 remains a strong extracted-H_f baseline: `True`
4. project-H_f-assisted limitation is reduced because the extracted path uses real Sionna channel tensors for H_f construction.
5. full native-only benchmark completed: `False`.

Current interpretation remains native-channel-assisted plus native-receiver-assisted, not full native-only.
