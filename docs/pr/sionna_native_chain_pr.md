# PR Title Suggestion

`feat: add Sionna-native OFDM receiver chain with learned beamformers`

## Summary

This PR adds an optional Sionna-native OFDM receiver-chain integration branch on top of the existing benchmark and learned-training work. The scope is synthetic, channel-level OFDM only. It validates a real Sionna receiver path for analytic and learned beamformer insertion without changing the released `v0.1.0`, `v0.2.0`, or `v0.3.0` claims.

## What Is Included

- native OFDM baseline chain
- pilot pattern audit
- estimator/equalizer minimal demo
- beamformed receiver shape bridge
- project `RZF` / `WMMSE-iter` insertion into the native receiver chain
- learned `residual_rzf` insertion
- learned `residual_wmmse_distill` insertion
- native learned comparison
- lightweight SNR mini benchmark
- native-chain artifact manifest
- minimal reviewer reproduction command

## What Is Explicitly Not Included

- Sionna RT
- ray tracing
- 5G NR full stack
- full native-only benchmark
- production e2e
- stable learned `> WMMSE-iter5` claim

## Key Results

- The native receiver path now succeeds for `no_precoding`, `project_rzf`, `project_wmmse_iter_2`, `project_wmmse_iter_5`, `learned_residual_rzf`, and `learned_residual_wmmse_distill`.
- Both learned methods run through real Sionna `channel` / `estimator` / `equalizer` / `demapper` components.
- `teacher_used_during_inference = false` remains preserved for learned inference.
- The learned methods stay close to the analytic baselines in the current synthetic/project-H_f-assisted native receiver benchmark.

## Teacher Leakage Statement

- `teacher_used_during_inference = false` for the learned native-chain runs
- no online `WMMSE` teacher call is used during learned inference
- `residual_wmmse_distill` remains a checkpoint-only learned inference path

## Project-H_f-Assisted Limitation

The receiver path is genuinely Sionna-native, but the current benchmark still consumes project-side frequency-domain `H_f` / precoder interfaces. This should be described as a synthetic/project-H_f-assisted native receiver benchmark, not as a full native-only benchmark.

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- current branch result after this phase should remain green

## Reproduction Commands

```bash
python scripts/reproduce_sionna_native_chain_minimal.py \
  --out outputs/repro/sionna_native_chain_minimal_summary.json

python scripts/generate_sionna_native_chain_artifact_manifest.py \
  --out outputs/sionna_native_chain/native_chain_artifact_manifest.json
```

## Risk Assessment

- moderate integration risk because the receiver path depends on optional Sionna APIs and shape conventions
- low scope risk because the branch still excludes RT, ray tracing, and 5G NR full stack
- moderate interpretation risk if readers overstate the native receiver benchmark as a full native-only system

## Merge Recommendation

Merge if the project wants a documented optional Sionna-native receiver-chain branch that cleanly supports analytic and learned beamformer insertion. Keep `residual_rzf` as the clean mainline interpretation, and keep `residual_wmmse_distill` as a valid secondary result without overclaiming a universal win over `WMMSE-iter5`.
