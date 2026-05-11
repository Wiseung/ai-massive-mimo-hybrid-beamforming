# PR Title Suggestion

`feat: add Sionna OFDM learned beamformer training pipeline`

## PR Summary

This PR adds an optional Sionna-compatible OFDM learned-training branch on top of the existing PyTorch beamforming benchmark. The scope is synthetic OFDM link-level training only. It extends the earlier optional Sionna PHY/OFDM smoke path into a compact training/evaluation workflow without changing the released `v0.1.0` and `v0.2.0` claims.

## What Is Included

- TinyNeuralBeamformer training baseline
- Residual-RZF learned refinement
- Unfolded-Lite learned model
- Residual WMMSE-distilled model
- evaluation against `RZF` and `WMMSE-iter5`
- multi-seed quick robustness benchmark
- latency / parameter benchmark
- residual correction analysis
- quick scale sweep
- quick train-SNR ablation
- distillation-weight sweep
- teacher leakage audit
- learned-training artifact manifest
- minimal reviewer smoke reproduction command

## What Is Explicitly Not Included

- Sionna RT
- ray tracing
- 5G NR full stack
- production e2e training
- learned model surpassing `WMMSE-iter5`

## Key Results

- TinyNeuralBeamformer is much weaker than analytic priors.
- Residual-RZF reaches `RZF`-level performance with much lower inference latency than `WMMSE-iter5`.
- Residual WMMSE-distillation is safe but gives only a tiny improvement over Residual-RZF.
- No learned model beats `WMMSE-iter5`.

## Negative / Near-null Result

`WMMSE` distillation gives only a tiny improvement over the plain residual-RZF model:

- Residual-RZF: `gap_to_wmmse_iter_5 = -0.4999%`
- Residual WMMSE-distill: `gap_to_wmmse_iter_5 = -0.4997%`

This should not be presented as a distillation breakthrough.

## Teacher Leakage Audit

- `teacher_used_during_training = true`
- `teacher_used_during_inference = false`
- `model_forward_calls_wmmse = false`
- `evaluate` uses `WMMSE` only as a baseline
- `leakage_detected = false`

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- current branch result: `45 passed`

## How To Reproduce

```bash
python scripts/reproduce_sionna_training_minimal.py \
  --out outputs/repro/sionna_training_minimal_summary.json

python scripts/generate_sionna_training_artifact_manifest.py \
  --out outputs/sionna_training_artifact_manifest.json
```

## Risks

- moderate maintenance risk because the optional Sionna training path spans several scripts and artifact generators
- low benchmark risk because the branch preserves the optional dependency boundary
- moderate interpretation risk if readers overstate the WMMSE-distillation result; release wording should keep it near-null

## Merge Recommendation

Merge if the project wants a documented optional Sionna learned-training branch heading toward a possible `v0.3.0`. Keep `Residual-RZF` as the clean mainline interpretation, and retain the WMMSE-distilled result as a safe but near-null extension.
