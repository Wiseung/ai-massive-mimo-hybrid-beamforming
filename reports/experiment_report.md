# Experiment Report

## Background

This project studies AI-assisted hybrid beamforming and precoding for massive MIMO and mmWave downlink systems. The primary engineering objective is not maximum model complexity, but a reproducible baseline pipeline that can run end-to-end on a single RTX 5090 24GB GPU.

## System Model

The current validated setup is a multi-user downlink MISO configuration with:

- BS antenna count `Nt`
- user count `K`
- one stream per user
- total transmit power normalized to `1`
- spectral efficiency evaluated through achievable sum-rate

For hybrid methods, the full precoder is composed as:

```text
F = F_RF F_BB
```

with soft power normalization and constant-modulus projection.

## Dataset

### Synthetic

Implemented synthetic channel families:

- IID Rayleigh narrowband channels
- sparse geometric mmWave narrowband channels
- simple wideband OFDM-like channels

The main validated benchmark used sparse geometric mmWave channels.

### DeepMIMO

DeepMIMO loading hooks were added, but the local dataset is not present.

DeepMIMO experiments have not been run because the dataset is not present locally.

## Baselines

Implemented baselines:

- MRT
- ZF
- RZF
- DFT codebook hybrid precoding
- OMP-style sparse hybrid precoding

Observed behavior on the current synthetic benchmark:

- low SNR: MRT and RZF are stronger than ZF
- high SNR: ZF and RZF overtake MRT
- OMP and DFT are slower and weaker than digital ZF/RZF in the current lightweight setup

## AI Models

Implemented AI models:

- MLP beamformer
- CNN beamformer
- unfolded PGA scaffold with learnable step sizes

The current CNN model is intentionally small. It is meant to prove training, evaluation, checkpointing, AMP, and reproducibility before scaling model capacity.

## Training Setup

- framework: PyTorch
- device policy: `auto`
- AMP: enabled on CUDA, disabled on CPU
- logging: TensorBoard + CSV
- checkpointing: `best.pt`, `last.pt`
- loss: negative sum-rate with power and constant-modulus penalties

## Results

### Verified Synthetic Baseline SE vs SNR

| Method | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRT | 0.1730 | 0.5137 | 1.3942 | 3.2504 | 6.2666 | 10.0521 | 13.8777 |
| ZF | 0.1012 | 0.3134 | 0.9323 | 2.5169 | 5.7204 | 10.5916 | 16.5179 |
| RZF | 0.1716 | 0.5041 | 1.3536 | 3.1933 | 6.4542 | 11.1204 | 16.7830 |
| DFT | 0.1349 | 0.3969 | 1.0586 | 2.4105 | 4.5415 | 7.1477 | 9.7124 |
| OMP | 0.1335 | 0.3927 | 1.0459 | 2.3745 | 4.4517 | 6.9579 | 9.3756 |

### Verified CNN Training Status

The synthetic CNN training pipeline completed and produced `best.pt`, `last.pt`, TensorBoard logs, CSV logs, and evaluation summaries. The current lightweight model remains below the stronger classical baselines, which is expected at this maturity level.

## Discussion

The strongest current engineering result is not AI superiority yet; it is a stable benchmark and training pipeline. RZF is currently the most reliable classical baseline across the tested SNR range. This gives a defensible reference before moving to larger CNNs, better hybrid parameterizations, or deeper unfolding.

## Limitations

- DeepMIMO experiments are not yet validated locally
- Sionna runtime demo is not yet validated locally
- the unfolded PGA model is only a minimal scaffold, not a tuned algorithmic reproduction
- no claim is made that the current AI models beat optimized classical baselines
- OMP is a lightweight approximation, not a full reference implementation from a dedicated hybrid precoding paper

## Future Work

1. Connect DeepMIMO `O1_28` using the current unified `deepmimo` package and validate channel shapes against one local scenario.
2. Add stronger hybrid-output AI models that explicitly predict `F_RF` and `F_BB`.
3. Expand evaluation with larger sample counts and deeper ablations.
4. Replace the unfolded PGA scaffold with a closer algorithmic update rule tied to the actual rate objective.
5. Install Sionna and validate the minimum differentiable end-to-end notebook.
