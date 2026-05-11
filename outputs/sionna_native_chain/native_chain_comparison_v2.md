# Sionna Native Chain Comparison

1. receiver chain succeeds under no_precoding: `False`
2. receiver chain succeeds under beamforming methods: `False`
3. beamforming chain truly uses Sionna ResourceGrid: `True`
4. beamforming chain truly uses Sionna OFDMChannel/ApplyOFDMChannel path: `False`
5. project_rzf improves no_precoding approximate sum-rate: `True`
6. project_wmmse_iter_5 improves project_rzf approximate sum-rate: `True`
7. current best method by approximate sum-rate: `project_wmmse_iter_5`
8. current learned-beamformer insertion recommendation: `True`
9. next step should connect residual_rzf learned model: `True`

## Fallback Notes
- Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable.
- Pilot-enabled ResourceGrid is required when --enable-receiver-chain is used.
- Project frequency-domain precoders are the current clean insertion path because they match H_f=(B,Nsc,K,Nt) directly.
- Optional Sionna RZFPrecoder is audited separately and recorded only as a shape-checked reference path.
