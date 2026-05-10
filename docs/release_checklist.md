# Release Checklist

## Environment

- Python `3.11+` recommended for the DeepMIMO path.
- PyTorch with CUDA support is recommended for the benchmark and latency scripts.
- DeepMIMO is optional for CI and required only for the local DeepMIMO benchmark path.

## Data Preparation

- Synthetic data:
  - `outputs/data/synthetic_narrowband.pt`
- DeepMIMO smoke tensor:
  - `outputs/data/deepmimo_asu_campus_3p5_narrowband.pt`
- Do not commit raw DeepMIMO scenario downloads or large training checkpoints to git.

## Required Commands

```bash
python -m compileall src scripts tests
pytest -q
python scripts/generate_artifact_manifest.py --out outputs/artifact_manifest.json
python scripts/reproduce_minimal.py --out outputs/repro/minimal_repro_summary.json
```

Optional local quality checks:

```bash
ruff check src scripts tests
black --check src scripts tests
```

## Key Artifacts

- `outputs/artifact_manifest.json`
- `outputs/artifact_manifest.md`
- `outputs/comparisons/model_families_v4/model_family_table.csv`
- `outputs/comparisons/latency_v2/latency_table.csv`
- `outputs/comparisons/unfolded_wmmse_lite_sweep/best_variant.yaml`
- `outputs/comparisons/deepmimo_model_family_random_vs_contiguous.csv`

## Consistency Checks

- Artifact manifest commit field must match the current `git rev-parse HEAD` at generation time.
- DeepMIMO benchmark summaries must continue to state `K=4, Nt=8, Nsc=1`.
- `residual_wmmse` must remain documented as RZF-level after teacher leakage fix.
- `unfolded_wmmse_lite` latency hotspot must remain attributed to the WMMSE initializer, not the learned refinement.

## Known Limitations

- Current DeepMIMO scope is a filtered `asu_campus_3p5` benchmark with `K=4, Nt=8, Nsc=1`.
- No wideband benchmark is claimed.
- No Sionna end-to-end benchmark is claimed.
- Best current WMMSE-lite variant matches `wmmse_iter_5` in SE but not in latency.
- Hybrid analog / RF constrained training is not the primary validated path in the current release.

## Claims That Must Not Be Made

- `Nt=64` DeepMIMO benchmark completed
- wideband benchmark completed
- Sionna end-to-end benchmark completed
- WMMSE-lite is a lower-latency replacement for `wmmse_iter_5`
- `residual_wmmse` exceeds WMMSE after the teacher leakage fix
