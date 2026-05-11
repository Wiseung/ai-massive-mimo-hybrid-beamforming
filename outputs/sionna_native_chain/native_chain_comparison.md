# Sionna Native Chain Comparison

1. Beamforming chain truly uses Sionna ResourceGrid: `True`
2. Beamforming chain truly uses Sionna OFDMChannel/ApplyOFDMChannel path: `False`
3. Beamforming chain truly uses Sionna estimator/equalizer/demapper: `False`
4. Fallback was needed in beamforming chain: `True`
5. project_rzf improves baseline chain symbol MSE proxy: `False`
6. project_wmmse_iter_5 improves project_rzf approximate sum-rate: `True`
7. Current best method by approximate sum-rate: `project_wmmse_iter_5`
8. Learned beamformer insertion point recommendation: `True`

## Fallback Notes
- Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable.
- Project frequency-domain precoders are the current clean insertion path because they match H_f=(B,Nsc,K,Nt) directly.
- Optional Sionna RZFPrecoder is audited separately and recorded only as a shape-checked reference path.
