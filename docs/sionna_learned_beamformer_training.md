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

## Robustness, Latency, and Ablation

Quick multi-seed benchmark (`seeds = 1, 2, 3`):

- `SionnaOFDMResidualRZFBeamformer` remains the strongest learned method:
  `mean_sum_rate_mean = 17.656879 +/- 0.020110`
  `gap_to_rzf_mean = +0.0259% +/- 0.2038%`
  `gap_to_wmmse_iter_5_mean = -0.5732% +/- 0.1216%`
- `SionnaOFDMUnfoldedLiteBeamformer` remains below residual-RZF:
  `mean_sum_rate_mean = 17.354924 +/- 0.028487`
  `gap_to_rzf_mean = -0.7270% +/- 0.2053%`
- `TinyNeuralBeamformer` stays far below the communication-prior family:
  `mean_sum_rate_mean = 1.295672 +/- 0.006746`
  `gap_to_rzf_mean = -91.8285% +/- 0.0457%`
- This multi-seed result is from `--quick` mode and should not be presented as a full robustness study.

Latency and parameter benchmark (`B=128, Nsc=8, K=4, Nt=16`):

- `rzf`: `1.069 ms`, `0` params
- `tiny_neural_beamformer`: `0.848 ms`, `33,152` params
- `sionna_ofdm_residual_rzf`: `2.246 ms`, `66,049` params
- `sionna_ofdm_unfolded_lite`: `56.120 ms`, `198,147` params
- `wmmse_iter_2`: `58.532 ms`
- `wmmse_iter_5`: `192.120 ms`
- `SionnaOFDMUnfoldedLiteBeamformer` latency is close to `wmmse_iter_2`, which is expected because its inference path includes the `wmmse_iter_2` initializer before learned refinement.
- `SionnaOFDMResidualRZFBeamformer` keeps a much lower inference cost because it refines an `RZF` initializer and does not require `WMMSE` at inference.

Residual correction analysis:

- `mean_delta_norm_ratio = 0.052996`
- `mean_correction_angle_deg = 0.269179`
- `high_snr_delta_norm_ratio = 0.020542`
- `mean_relative_se_gain_over_rzf = +0.004854%`
- `alpha ~= 0.0979`
- The residual correction is small and becomes smaller at high SNR in the current full run.
- The observed `+0.0134%` full-eval gap to `RZF` is better described as noise-level refinement than as a durable RZF-beating gain.
- The accurate description is therefore `RZF-level learned refinement`, not `RZF-beating model`.

Quick scale sweep:

- Evaluated `num_subcarriers in {4, 8, 16}`, `num_users in {2, 4}`, `num_bs_ant in {8, 16, 32}`
- `Nt=32, Nsc=16` combinations were explicitly marked `skipped_due_to_runtime` in quick mode
- Residual-RZF average gap across evaluated settings:
  `gap_to_rzf = +0.000002%`
  `gap_to_wmmse_iter_5 = -0.399179%`
- The quick scale sweep supports the same operating-point conclusion:
  residual-RZF tracks RZF closely and does not fairly beat `WMMSE-iter5`.

Quick train-SNR ablation:

- Compared training sets:
  `low_mid [0, 5, 10]`
  `high_only [15, 20]`
  `mixed_default [0, 5, 10, 15, 20]`
  `wide [-5, 0, 5, 10, 15, 20]`
- `high_only` is marginally best in this quick run:
  `mean_gap_to_rzf = +0.1047%`
  `high_snr_gap_to_rzf = -0.0486%`
- However, the span across train-SNR sets is tiny:
  mean-gap span to RZF is about `2.87e-5`
  high-SNR-gap span is about `6.10e-5`
- This quick ablation does not support a strong claim that mixed training is best, nor does it justify curriculum training yet.

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
