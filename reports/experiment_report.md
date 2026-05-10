# Experiment Report

## Background

This project targets AI-assisted beamforming and precoding for massive MIMO and mmWave downlink systems, with engineering priorities ordered as reproducibility, benchmark fairness, and runnable single-GPU training. The current work deliberately postponed Sionna-centric end-to-end expansion until the synthetic benchmark, classical baselines, and learned-model evaluation were all on solid footing.

## System Model

The current validated setup is a multi-user downlink MISO configuration with:

- BS antenna count `Nt`
- user count `K`
- one stream per user
- total transmit power normalized to `1`
- spectral efficiency computed as MU downlink achievable sum-rate

Hybrid precoder composition follows:

```text
F = F_RF F_BB
```

The learned models used in the verified synthetic experiments are digital-only, because that is the most stable route for fair comparison against MRT, ZF, and RZF teachers.

## Dataset

### Synthetic

Implemented synthetic channel families:

- IID Rayleigh narrowband channels
- sparse geometric mmWave narrowband channels
- simple OFDM-like channels

The verified benchmark in this report uses the sparse geometric mmWave narrowband dataset with:

- `10000` samples
- `64` BS antennas
- `4` users
- SNR grid `[-10, -5, 0, 5, 10, 15, 20]`

### DeepMIMO

A DeepMIMO v4 adapter and smoke path were added. The expected install command is now:

```bash
pip install deepmimo
```

DeepMIMO experiments have not been run locally in this session because the `deepmimo` package is not installed.

## Baselines

Implemented baselines:

- MRT
- ZF
- RZF
- DFT codebook hybrid precoding
- OMP-style sparse hybrid precoding

Verified synthetic baseline values:

| Method | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRT | 0.1656 | 0.5158 | 1.3711 | 3.1612 | 6.2269 | 10.1118 | 13.8191 |
| ZF | 0.1011 | 0.2970 | 0.8679 | 2.4394 | 5.6202 | 10.6454 | 16.1630 |
| RZF | 0.1643 | 0.5053 | 1.3317 | 3.1028 | 6.3026 | 11.1583 | 16.4744 |
| DFT | 0.1294 | 0.3962 | 1.0496 | 2.3661 | 4.5415 | 7.1965 | 9.7844 |

RZF remains the strongest overall reference baseline in the verified synthetic setup.

## AI Models

Implemented AI models:

- MLP beamformer
- CNN beamformer
- unfolded PGA scaffold

The main improvement in this round was not introducing a bigger novel model, but fixing the evaluation contract and adding a two-stage teacher-guided training path.

### Original CNN

The original CNN pipeline was runnable but weak once evaluated fairly:

- `mean_se = 0.5567`
- `mean_relative_gap_to_rzf = -0.9166`

This established that the earlier checkpoint was not competitive under a baseline-comparable evaluation.

### Warm-Started CNN

The improved training path uses:

1. supervised pretraining to an `RZF` teacher
2. rate-based fine-tuning from the pretrained checkpoint

This changed the fair synthetic benchmark result to:

- `mean_se = 5.0524`
- `mean_relative_gap_to_rzf = -0.0272`
- `mean_relative_gap_to_best_baseline = -0.0383`

Per-SNR CNN values after warm-start:

| Method | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.1656 | 0.5158 | 1.3710 | 3.1609 | 6.2266 | 10.1113 | 13.8154 |

The improved CNN is close to MRT and RZF through low and medium SNR, but it still trails RZF at higher SNR. The repository does not claim learned superiority over RZF on this benchmark.

## Training Setup

- framework: PyTorch
- device policy: `auto`
- GPU used in practice: single RTX 5090 Laptop GPU
- AMP: enabled on CUDA, disabled on CPU
- checkpointing: `best.pt`, `last.pt`
- logs: TensorBoard + CSV
- fair evaluation split: deterministic validation subset reused across learned and baseline evaluation

### Two-Stage Training

Stage A, supervised warm-start:

```text
loss_pretrain = ||F_pred - F_teacher||_F^2
```

Stage B, rate fine-tuning:

```text
loss = -mean(sum_rate) + lambda_power * power_violation + lambda_const * constant_modulus_violation
```

### Logged Metrics

Training logs now include:

- sum_rate
- power_violation
- constant_modulus_violation
- precoder_norm
- gradient_norm
- learning_rate

## Results

### Fair Evaluation

The evaluation path now reports:

- `mean_se`
- `se_by_snr`
- `relative_gap_to_rzf`
- `relative_gap_to_best_baseline`

and the unified script `scripts/evaluate_all.py` produces one common CSV and one common SE-vs-SNR figure for baselines and learned models together.

### SNR Conditioning

An ablation between conditioned and non-conditioned warm-started CNNs shows almost no difference on the current synthetic benchmark:

- conditioned: `mean_se = 5.05235`
- not conditioned: `mean_se = 5.05239`

This suggests the dominant gain in this round came from teacher warm-start and fairer model design rather than from explicit SNR conditioning.

## Discussion

The most important correction in this round was methodological. The original learned result was not only weak; it was also not summarized in the same form as the baseline SE-vs-SNR benchmark. Once the evaluation contract was unified, the repository could cleanly show both the weakness of the original CNN and the effectiveness of the improved warm-start pipeline.

The final warm-started CNN is now a credible baseline learner. It is no longer collapsed, and it tracks the stronger classical methods closely across much of the tested range. However, RZF still remains the stronger method at higher SNR, so any claim that the learned model has surpassed classical baselines would be false.

## Limitations

- DeepMIMO v4 runtime has not been validated locally because `deepmimo` is not installed here
- no real DeepMIMO benchmark results are reported
- Sionna runtime remains optional and has not been validated in this session
- the unfolded PGA model is still a scaffold, not a tuned unfolding baseline
- the current AI models are still digital-only in the verified experiments

## Future Work

1. Install `deepmimo` and run the `asu_campus_3p5` smoke path plus baseline benchmark.
2. Extend teacher-guided training to stronger hybrid-output parameterizations.
3. Investigate why the learned model still falls behind RZF at high SNR.
4. Add a stronger analog/digital factorized learned model after the current fair benchmark is stable.
5. Revisit Sionna later for differentiable end-to-end links after DeepMIMO experiments are actually running.
