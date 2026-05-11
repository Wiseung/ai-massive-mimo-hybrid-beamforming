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

Train:

```bash
python scripts/train_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_learned_beamformer.yaml \
  --out outputs/runs/sionna_ofdm_learned_beamformer
```

Evaluate:

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

## Notes

- The current implementation uses `TinyNeuralBeamformer` as an experimental learned baseline.
- Real Sionna `ResourceGrid` and PHY AWGN are used when available; explicit fallback is recorded in logs and summaries.
- This pipeline does not modify the published `v0.1.0` or `v0.2.0` benchmark claims.

## Future Work

- replace `TinyNeuralBeamformer` with `residual_rzf` or `unfolded_wmmse_lite`
- Sionna-native equalizer/detector chain
- DeepMIMO-to-Sionna comparison
- optional Sionna RT channel generation after the PHY path is stable
