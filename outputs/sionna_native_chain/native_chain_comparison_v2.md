# Sionna Native Chain Comparison

1. receiver chain succeeds under no_precoding: `True`
2. receiver chain succeeds under beamforming methods: `True`
3. beamforming chain truly uses Sionna ResourceGrid: `True`
4. beamforming chain truly uses Sionna OFDMChannel/ApplyOFDMChannel path: `True`
5. project_rzf improves no_precoding approximate sum-rate: `False`
6. project_wmmse_iter_5 improves project_rzf approximate sum-rate: `True`
7. current best method by approximate sum-rate: `project_wmmse_iter_5`
8. current learned-beamformer insertion recommendation: `True`
9. next step should connect residual_rzf learned model: `True`

## Receiver Status
- native receiver success methods: `['no_precoding', 'project_rzf', 'project_wmmse_iter_2', 'project_wmmse_iter_5']`
- fallback-only methods: `[]`
- first native failure stage: ``
- first native failure reason: ``

## Fallback Notes
- Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable.
- Pilot-aware native beamformed receiver mode uses num_tx=1, num_streams_per_tx=K, rx_tx_association=ones(K,1), and at least one non-pilot OFDM symbol.
- receiver-mode=proxy keeps project-side proxy metrics only.
- receiver-mode=native requires a real Sionna receiver path and records exact failure stage/reason if it fails.
- receiver-mode=auto attempts the native receiver first and falls back to proxy metrics if needed.
