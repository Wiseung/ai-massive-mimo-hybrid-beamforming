# Native Learned Beamforming Comparison

1. learned_residual_rzf native receiver success: `True`
2. learned_residual_wmmse_distill native receiver success: `True`
3. learned_residual_rzf gap to project_rzf: `+0.692277%`
4. learned_residual_rzf gap to project_wmmse_iter_5: `+3.742379%`
5. learned_residual_wmmse_distill gap to project_rzf: `+1.836184%`
6. learned_residual_wmmse_distill gap to project_wmmse_iter_5: `+4.920937%`
7. teacher_used_during_inference false for learned methods: `True`
8. recommend learned_residual_rzf as native-chain mainline: `True`
9. current result remains synthetic/project-H_f-assisted native receiver benchmark: `True`

## Notes
- analytic native receiver success methods from prior pure-analytic run: `['no_precoding', 'project_rzf', 'project_wmmse_iter_2', 'project_wmmse_iter_5']`
- learned native receiver success methods: `['no_precoding', 'project_rzf', 'project_wmmse_iter_2', 'project_wmmse_iter_5', 'learned_residual_rzf', 'learned_residual_wmmse_distill']`
- learned skipped missing checkpoint: `[]`
- gap calculations are referenced to the analytic methods inside the same learned native-chain run, not to a separate earlier analytic artifact.
