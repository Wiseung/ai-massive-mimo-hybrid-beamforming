# Sionna Learned Beamformer Training

This branch adds an optional multi-SNR OFDM learned beamformer training pipeline on synthetic channels. It is a link-level experiment for coexistence with the current PyTorch beamforming codebase, not a production end-to-end stack.

## Scope

- multi-SNR OFDM learned beamformer training
- optional dependency: `sionna-no-rt`
- synthetic OFDM channel only
- no Sionna RT
- no ray tracing
- no full 5G NR stack
- not a production e2e training pipeline

## Commands

Train tiny baseline:

```bash
python scripts/train_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_learned_beamformer.yaml \
  --out outputs/runs/sionna_ofdm_learned_beamformer
```

Evaluate tiny baseline:

```bash
python scripts/evaluate_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_learned_beamformer.yaml \
  --ckpt outputs/runs/sionna_ofdm_learned_beamformer/best.pt \
  --out outputs/comparisons/sionna_ofdm_learned_beamformer
```

Smoke:

```bash
python scripts/train_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_learned_beamformer.yaml \
  --out outputs/runs/sionna_ofdm_learned_beamformer_smoke \
  --smoke
```

## Metrics

- SE vs SNR
- MSE vs SNR
- gap to RZF
- gap to WMMSE-iter5
- power violation

## Current Results

TinyNeuralBeamformer full result:

- `learned_mean_sum_rate = 8.723585`
- `mean_gap_to_rzf = -39.5834%`
- `mean_gap_to_wmmse_iter_5 = -39.9628%`
- high-SNR gap to RZF gets worse from `-47.5942%` at `10 dB` to `-65.9680%` at `20 dB`

SionnaOFDMResidualRZF full result:

- `learned_mean_sum_rate = 17.657597`
- `mean_gap_to_rzf = +0.0134%`
- `mean_gap_to_wmmse_iter_5 = -0.4999%`
- mean high-SNR gap to RZF: `-0.0079%`
- mean high-SNR gap to WMMSE-iter5: `-0.1485%`

SionnaOFDMUnfoldedLite full result:

- `init_method = wmmse_iter_2`
- `num_layers = 3`
- `learned_mean_sum_rate = 17.466006`
- `mean_gap_to_rzf = -0.3550%`
- `mean_gap_to_wmmse_iter_5 = -0.8715%`
- mean high-SNR gap to RZF: `-1.4279%`
- mean high-SNR gap to WMMSE-iter5: `-1.5685%`

Interpretation:

- Communication-prior models improve over `TinyNeuralBeamformer` by a large margin.
- `SionnaOFDMResidualRZFBeamformer` is the strongest current learned method on this synthetic OFDM setup.
- `SionnaOFDMUnfoldedLiteBeamformer` is also strong, but under the current configuration it remains slightly below residual-RZF and is slower to train because the `wmmse_iter_2` initializer is evaluated per subcarrier.
- Neither learned method fairly exceeds `WMMSE-iter5`; that remaining gap is reported directly.

## Notes

- `TinyNeuralBeamformer` remains the unconstrained learned baseline, but it is clearly too weak at high SNR.
- `SionnaOFDMResidualRZFBeamformer` uses an RZF prior plus learnable residual correction and does not depend on WMMSE at inference.
- `SionnaOFDMUnfoldedLiteBeamformer` uses a configurable few-iteration initializer and learnable refinement layers; if the initializer is `wmmse_iter_k`, that is reported explicitly in summaries.
- Real Sionna `ResourceGrid` and PHY AWGN are used when available; explicit fallback is recorded in logs and summaries.
- This pipeline does not modify the published `v0.1.0` or `v0.2.0` benchmark claims.

## Future Work

- use `SionnaOFDMResidualRZFBeamformer` as the main learned baseline for the next phase
- revisit `SionnaOFDMUnfoldedLiteBeamformer` with different `init_method` or lower-cost initialization
- Sionna-native equalizer/detector chain
- DeepMIMO-to-Sionna comparison
- optional Sionna RT channel generation after the PHY path is stable
