# AI Massive MIMO Hybrid Beamforming

Reproducible PyTorch-based single-GPU project for AI massive MIMO hybrid beamforming and precoding. The current implementation is optimized for the most stable path first:

- Synthetic CSI generation
- Sum-rate and spectral-efficiency metrics
- Classical MRT/ZF/RZF/DFT/OMP baselines
- Lightweight CNN/MLP beamformers
- Optional DeepMIMO and Sionna integration hooks

The repository is designed around a single RTX 5090 24GB GPU, but it also runs on CPU for smoke checks.

## Project Structure

```text
src/beamforming/
scripts/
configs/
tests/
outputs/
reports/
notebooks/
```

## Environment Installation

Recommended Python: `3.10` to `3.14`. This repository was validated with local `Python 3.13.9` and `torch 2.11.0+cu130`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

PyTorch installation should be selected from the official compatibility selector instead of hard-coding an old command:

- [PyTorch local install guide](https://pytorch.org/get-started/locally/)

Optional dependencies:

```bash
pip install sionna
pip install --pre deepmimo
```

References:

- [Sionna documentation](https://nvlabs.github.io/sionna/index.html)
- [DeepMIMO documentation](https://www.deepmimo.net/docs/index.html)

## RTX 5090 24GB Recommended Config

- Device: single CUDA GPU
- Synthetic training batch size: `256`
- Synthetic dataset size for quick iteration: `10000`
- Mixed precision: enabled automatically on CUDA
- Start with `configs/synthetic_cnn.yaml`

If you are using this same machine, `scripts/check_env.py` should report:

- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- CUDA available: `True`

## Synthetic Quick Start

1. Check environment:

```bash
python scripts/check_env.py
python scripts/smoke_test.py
pytest -q
```

2. Generate synthetic CSI:

```bash
python scripts/make_synthetic_csi.py \
  --out outputs/data/synthetic_narrowband.pt \
  --num-samples 10000 \
  --num-bs-ant 64 \
  --num-users 4 \
  --num-paths 3
```

3. Run baselines:

```bash
python scripts/run_baselines.py \
  --data outputs/data/synthetic_narrowband.pt \
  --methods mrt zf rzf dft omp \
  --out outputs/runs/baselines_synthetic
```

4. Train CNN beamformer:

```bash
python scripts/train.py \
  --config configs/synthetic_cnn.yaml \
  --data outputs/data/synthetic_narrowband.pt \
  --out outputs/runs/cnn_synthetic
```

5. Evaluate CNN beamformer:

```bash
python scripts/evaluate.py \
  --config configs/synthetic_cnn.yaml \
  --data outputs/data/synthetic_narrowband.pt \
  --ckpt outputs/runs/cnn_synthetic/best.pt \
  --out outputs/runs/cnn_synthetic_eval
```

## Synthetic Dataset

Supported synthetic channel types:

- `rayleigh`: IID narrowband Rayleigh fading
- `mmwave`: sparse geometric narrowband mmWave channel
- `wideband`: simple OFDM-like frequency-selective channel

Default parameter intent:

- `num_bs_ant: 64`
- `num_users: 4`
- `num_rf_chains: 4`
- `num_paths: 3`
- `snr_db: [-10, -5, 0, 5, 10, 15, 20]`
- `num_samples: 50000`

## Baselines

Implemented baselines:

- MRT
- ZF
- RZF
- DFT codebook hybrid precoding
- OMP-style sparse hybrid precoding

All baselines share the same input and output contract:

- Input: complex channel tensor `H`
- Output: `precoder`, `sum_rate`, `se`, `runtime_sec`

Generated artifacts:

- `se_vs_snr.png`
- `sum_rate_vs_users.png`
- `sum_rate_vs_rf_chains.png`
- `runtime_comparison.png`
- `metrics/baseline_results.csv`

## Training

Implemented models:

- `MLPBeamformer`
- `CNNBeamformer`
- `UnfoldedPGABeamformer` minimal scaffold

Training features:

- AMP on CUDA, auto-disabled on CPU
- `best.pt` and `last.pt`
- resume from checkpoint
- TensorBoard logs
- CSV logs

Loss:

```text
loss = -mean(sum_rate) + lambda_power * power_violation + lambda_const * constant_modulus_violation
```

## DeepMIMO Data Preparation

This repository does not fabricate DeepMIMO results. If local data is missing, scripts fail explicitly with:

```text
DeepMIMO experiments have not been run because the dataset is not present locally.
```

Recommended path to integrate `O1_28`:

1. Install the current package:

```bash
pip install --pre deepmimo
```

2. Follow the current DeepMIMO tutorials and docs for dataset download and generation:

- [DeepMIMO tutorials](https://deepmimo.net/docs/tutorials/index.html)
- [DeepMIMO docs](https://www.deepmimo.net/docs/index.html)

3. Produce either:

- a local DeepMIMO scenario directory loadable by `deepmimo.load(...)`, or
- a saved `.pt` / `.npy` channel tensor with shape adaptable to `(N, K, Nt)`

4. Inspect the dataset:

```bash
python scripts/inspect_deepmimo.py --scenario-path /path/to/O1_28
```

5. Run baselines on DeepMIMO:

```bash
python scripts/run_baselines.py \
  --dataset-type deepmimo \
  --data /path/to/O1_28 \
  --methods mrt zf rzf dft omp \
  --out outputs/runs/baselines_deepmimo
```

6. Train on DeepMIMO:

```bash
python scripts/train.py \
  --dataset-type deepmimo \
  --config configs/deepmimo_cnn.yaml \
  --data /path/to/O1_28 \
  --out outputs/runs/cnn_deepmimo
```

DeepMIMO experiments have not been run because the dataset is not present locally.

## Sionna Demo

Sionna is treated as an optional dependency. The minimum demo artifact is:

- [notebooks/02_sionna_e2e_demo.ipynb](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/notebooks/02_sionna_e2e_demo.ipynb)

The implementation target is a minimal differentiable link with:

- Sionna-generated MIMO channel
- learnable beamforming block
- negative achievable rate or BLER proxy loss

Current status:

- Notebook scaffold exists
- Python hook exists in `src/beamforming/data/sionna_generator.py`
- Runtime validation is blocked because `sionna` is not installed locally

## Reproducibility

- Fixed seeds in scripts and trainer configs
- Script-generated metrics and figures
- Deterministic unit tests for core math and rate behavior

## Current Verified Results

On the local synthetic mmWave dataset with `1024` samples, `64` BS antennas, and `4` users, the verified baseline SE vs SNR trend is:

| Method | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRT | 0.1730 | 0.5137 | 1.3942 | 3.2504 | 6.2666 | 10.0521 | 13.8777 |
| ZF | 0.1012 | 0.3134 | 0.9323 | 2.5169 | 5.7204 | 10.5916 | 16.5179 |
| RZF | 0.1716 | 0.5041 | 1.3536 | 3.1933 | 6.4542 | 11.1204 | 16.7830 |
| DFT | 0.1349 | 0.3969 | 1.0586 | 2.4105 | 4.5415 | 7.1477 | 9.7124 |
| OMP | 0.1335 | 0.3927 | 1.0459 | 2.3745 | 4.4517 | 6.9579 | 9.3756 |

These are script-generated results from `outputs/runs/baselines_synthetic/metrics/baseline_results.csv`.

## Final Acceptance Commands

```bash
python -m compileall src scripts tests
pytest -q
python scripts/check_env.py
python scripts/smoke_test.py
python scripts/make_synthetic_csi.py --out outputs/data/synthetic_narrowband.pt --num-samples 10000 --num-bs-ant 64 --num-users 4 --num-paths 3
python scripts/run_baselines.py --data outputs/data/synthetic_narrowband.pt --methods mrt zf rzf dft --out outputs/runs/baselines_synthetic
python scripts/train.py --config configs/synthetic_cnn.yaml --data outputs/data/synthetic_narrowband.pt --out outputs/runs/cnn_synthetic
python scripts/evaluate.py --config configs/synthetic_cnn.yaml --data outputs/data/synthetic_narrowband.pt --ckpt outputs/runs/cnn_synthetic/best.pt --out outputs/runs/cnn_synthetic_eval
```
