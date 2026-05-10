# PR Title Suggestion

`Add optional Sionna PHY/OFDM smoke demos and differentiable beamformer checks`

## PR Summary

This PR adds an optional Sionna-compatible PHY/OFDM smoke path on top of the existing PyTorch beamforming benchmark repository. The goal is to validate coexistence with `sionna-no-rt`, confirm usable Sionna 2.x PHY/OFDM APIs, and prove that a tiny learned beamformer can backpropagate through an OFDM-style link without changing the released `v0.1.0` benchmark claims.

## What Is Included

- Sionna environment check
- Sionna PHY and OFDM API introspection
- PHY AWGN smoke demo
- OFDM ResourceGrid smoke demo
- OFDM beamforming bridge demo using project `RZF` / `WMMSE-iter5`
- Tiny differentiable beamformer
- Gradient check utility
- Few-step differentiable OFDM beamforming smoke demo
- Sionna-specific artifact manifest
- Optional tests only

## What Is Explicitly Not Included

- Sionna RT
- ray tracing
- full Sionna end-to-end training pipeline
- 5G NR full stack
- changes to `v0.1.0` benchmark conclusions
- mandatory Sionna dependency

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- current branch test result: `33 passed`
- differentiable demo:
  - `initial_loss = 2.0943`
  - `final_loss = 0.3303`
  - `loss_decreased = true`

## How To Reproduce

```bash
python scripts/check_sionna_env.py
python scripts/inspect_sionna_api.py --out outputs/sionna_smoke/sionna_api_summary.json
python scripts/inspect_sionna_ofdm_api.py --out outputs/sionna_smoke/sionna_ofdm_api_summary.json
python scripts/sionna_phy_awgn_demo.py --out outputs/sionna_smoke/sionna_phy_awgn_summary.json
python scripts/sionna_ofdm_resource_grid_demo.py --out outputs/sionna_smoke/sionna_ofdm_resource_grid_summary.json
python scripts/sionna_ofdm_beamforming_bridge_demo.py --out outputs/sionna_smoke/sionna_ofdm_beamforming_bridge_summary.json
python scripts/check_differentiable_beamformer_gradients.py --out outputs/sionna_smoke/differentiable_beamformer_gradcheck.json
python scripts/sionna_ofdm_differentiable_beamforming_demo.py --out outputs/sionna_smoke/sionna_ofdm_differentiable_beamforming_summary.json
python scripts/generate_sionna_artifact_manifest.py --out outputs/sionna_smoke/sionna_artifact_manifest.json
```

## Risk Assessment

- low risk to the released benchmark path because all Sionna code remains optional
- moderate maintenance risk if future Sionna APIs move or rename OFDM blocks
- low runtime risk on environments without Sionna because optional tests skip cleanly

## Merge Recommendation

Merge if the project wants a self-contained optional Sionna PHY/OFDM smoke track ahead of any heavier Sionna-native channel/equalizer or RT work.
