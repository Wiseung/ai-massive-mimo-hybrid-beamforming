# AI Massive MIMO Hybrid Beamforming

Reproducible PyTorch-based single-GPU project for massive MIMO / mmWave beamforming and precoding. The current repository is centered on a verified synthetic CSI pipeline and fair benchmark comparison before pushing further into heavier DeepMIMO or Sionna features.

## Current Status

- Synthetic pipeline is verified end-to-end.
- Classical baseline results are verified.
- The original CNN training path ran successfully but its fair evaluation was weak.
- A fair unified evaluation path now exists through `scripts/evaluate_all.py`.
- Teacher warm-start and SNR-conditioned training were added.
- Warm-started CNN now approaches the RZF baseline on the synthetic benchmark.
- High-SNR-weighted fine-tuning did not materially improve the synthetic high-SNR gap.
- Mixed-teacher warm-start produced only a marginal synthetic gain over the RZF teacher.
- DeepMIMO `v4` is installed locally and the `asu_campus_3p5` smoke path, baseline smoke benchmark, and a small learned smoke benchmark all ran successfully.
- Sionna remains optional and is not the mainline blocker for this repository.

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

Recommended Python: `3.11` to `3.14`. This repository was validated with local `Python 3.13.9` and `torch 2.11.0+cu130`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

PyTorch installation should still be selected from the current official compatibility selector:

- [PyTorch local install guide](https://pytorch.org/get-started/locally/)

Optional dependencies:

```bash
pip install deepmimo
pip install sionna
```

DeepMIMO notes:

- Prefer a Python `3.11+` environment for the DeepMIMO path.
- The DeepMIMO smoke test is independent from Sionna.
- Do not commit downloaded DeepMIMO scenarios or generated tensors into git.

References:

- [Sionna documentation](https://nvlabs.github.io/sionna/index.html)
- [DeepMIMO documentation](https://www.deepmimo.net/docs/index.html)

## RTX 5090 24GB Recommended Config

- Device: single CUDA GPU
- Synthetic training batch size: `256`
- Synthetic dataset size for quick iteration: `10000`
- Mixed precision: enabled automatically on CUDA
- Start with `configs/synthetic_cnn_pretrain.yaml` then `configs/synthetic_cnn_finetune.yaml`

On this machine, `scripts/check_env.py` reports:

- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- CUDA available: `True`

## Verified Synthetic Workflow

1. Environment checks:

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
  --methods mrt zf rzf dft \
  --out outputs/runs/baselines_synthetic
```

4. Supervised warm-start:

```bash
python scripts/pretrain.py \
  --config configs/synthetic_cnn_pretrain.yaml \
  --data outputs/data/synthetic_narrowband.pt \
  --teacher rzf \
  --out outputs/runs/cnn_pretrain_rzf
```

5. Rate fine-tuning:

```bash
python scripts/train.py \
  --config configs/synthetic_cnn_finetune.yaml \
  --data outputs/data/synthetic_narrowband.pt \
  --init-ckpt outputs/runs/cnn_pretrain_rzf/best.pt \
  --out outputs/runs/cnn_finetune_rzf
```

6. Fair evaluation of all methods:

```bash
python scripts/evaluate_all.py \
  --data outputs/data/synthetic_narrowband.pt \
  --ckpt outputs/runs/cnn_finetune_rzf/best.pt \
  --config configs/synthetic_cnn_finetune.yaml \
  --methods mrt zf rzf dft cnn \
  --out outputs/comparisons/synthetic_cnn_finetune
```

7. High-SNR and mixed-teacher ablations:

```bash
python scripts/train.py \
  --config configs/synthetic_cnn_finetune_highsnr.yaml \
  --data outputs/data/synthetic_narrowband.pt \
  --init-ckpt outputs/runs/cnn_pretrain_rzf/best.pt \
  --out outputs/runs/cnn_finetune_highsnr

python scripts/pretrain.py \
  --config configs/synthetic_cnn_pretrain_mixed_teacher.yaml \
  --data outputs/data/synthetic_narrowband.pt \
  --teacher mixed_rzf_zf \
  --out outputs/runs/cnn_pretrain_mixed_teacher

python scripts/train.py \
  --config configs/synthetic_cnn_finetune_mixed_teacher.yaml \
  --data outputs/data/synthetic_narrowband.pt \
  --init-ckpt outputs/runs/cnn_pretrain_mixed_teacher/best.pt \
  --out outputs/runs/cnn_finetune_mixed_teacher

python scripts/compare_ablation.py \
  --runs outputs/comparisons/synthetic_cnn_finetune \
         outputs/comparisons/synthetic_cnn_highsnr \
         outputs/comparisons/synthetic_cnn_mixed_teacher \
  --out outputs/comparisons/ablation
```

## Fair Evaluation Contract

Baseline and learned evaluation now use the same:

- validation split
- channel tensor subset
- SNR grid
- transmit power normalization
- noise variance definition
- channel normalization
- MU-MISO sum-rate definition

The fair learned-model summary is written by `scripts/evaluate.py` as:

- `mean_se`
- `se_by_snr`
- `mean_relative_gap_to_rzf`
- `mean_relative_gap_to_best_baseline`

Unified baseline + learned comparisons are written by `scripts/evaluate_all.py` to:

- `outputs/comparisons/.../*_all_methods.csv`
- `outputs/comparisons/.../*_se_vs_snr.png`

## Synthetic Results

### Baselines

Verified synthetic baseline SE / sum-rate values:

| Method | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRT | 0.1656 | 0.5158 | 1.3711 | 3.1612 | 6.2269 | 10.1118 | 13.8191 |
| ZF | 0.1011 | 0.2970 | 0.8679 | 2.4394 | 5.6202 | 10.6454 | 16.1630 |
| RZF | 0.1643 | 0.5053 | 1.3317 | 3.1028 | 6.3026 | 11.1583 | 16.4744 |
| DFT | 0.1294 | 0.3962 | 1.0496 | 2.3661 | 4.5415 | 7.1965 | 9.7844 |

### Original CNN

The original trained CNN was runnable but weak under fair evaluation:

- `mean_se = 0.5567`
- `mean_relative_gap_to_rzf = -0.9166`

This confirms the earlier issue was not only training instability, but also that the prior evaluation output was not a fair baseline-comparable benchmark summary.

### Warm-Started CNN

With `RZF teacher` pretraining and rate fine-tuning:

- `mean_se = 5.0524`
- `mean_relative_gap_to_rzf = -0.0272`
- `mean_relative_gap_to_best_baseline = -0.0383`

Per-SNR CNN values after warm-start:

| Method | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.1656 | 0.5158 | 1.3710 | 3.1609 | 6.2266 | 10.1113 | 13.8154 |

This model is close to MRT / RZF over most of the tested range, but it does not beat RZF at high SNR. The repository does not claim otherwise.

### SNR Conditioning Ablation

The current synthetic benchmark shows that SNR conditioning and no-SNR-conditioning are nearly identical once warm-start is present:

- with SNR conditioning: `mean_se = 5.05235`
- without SNR conditioning: `mean_se = 5.05239`

On this benchmark, the main gain came from fair evaluation plus teacher warm-start, not from a large SNR-conditioning effect.

### High-SNR Gap Status

The current high-SNR gap is still the main remaining issue on the synthetic benchmark:

- `RZF warm-start + SNR conditioning`: `gap_10db = -1.21%`, `gap_15db = -9.38%`, `gap_20db = -16.14%`
- `high-SNR weighted fine-tune`: `gap_10db = -1.22%`, `gap_15db = -9.38%`, `gap_20db = -16.14%`
- `mixed teacher`: `gap_10db = -1.21%`, `gap_15db = -9.38%`, `gap_20db = -16.13%`

This means the added weighting and mixed-teacher strategies only produced marginal changes. The repository does not claim that the high-SNR issue has been solved.

### Ablation Table

The current ablation table is generated by:

- `outputs/comparisons/ablation/ablation_table.csv`
- `outputs/comparisons/ablation/ablation_se_vs_snr.png`

## DeepMIMO v4

Current expected installation:

```bash
pip install deepmimo
```

DeepMIMO v4 quickstart shape assumed by this repository:

```python
import deepmimo as dm
dm.download("asu_campus_3p5")
dataset = dm.load("asu_campus_3p5")
channels = dataset.compute_channels()  # [n_ue, n_rx, n_tx, n_sub]
```

Supported loader entrypoints now include:

- `--scenario asu_campus_3p5`
- `--download`
- `--num-users`
- `--num-bs-ant`
- `--num-subcarriers`
- `--narrowband`

DeepMIMO inspection:

```bash
python scripts/inspect_deepmimo.py \
  --scenario asu_campus_3p5 \
  --download \
  --num-users 4 \
  --narrowband
```

If the package is missing, the script now reports:

```text
DeepMIMO is not installed. Install it with: pip install deepmimo
```

DeepMIMO baseline command:

```bash
python scripts/run_baselines.py \
  --dataset-type deepmimo \
  --scenario asu_campus_3p5 \
  --download \
  --methods mrt zf rzf dft \
  --out outputs/runs/baselines_deepmimo
```

Verified local DeepMIMO commands:

```bash
python scripts/inspect_deepmimo.py \
  --scenario asu_campus_3p5 \
  --download \
  --num-users 4 \
  --narrowband

python scripts/run_baselines.py \
  --dataset-type deepmimo \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --methods mrt zf rzf dft \
  --out outputs/runs/baselines_deepmimo_asu

python scripts/pretrain.py \
  --dataset-type deepmimo \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --config configs/deepmimo_cnn_pretrain.yaml \
  --teacher rzf \
  --out outputs/runs/deepmimo_cnn_pretrain_rzf

python scripts/train.py \
  --dataset-type deepmimo \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --config configs/deepmimo_cnn_finetune.yaml \
  --init-ckpt outputs/runs/deepmimo_cnn_pretrain_rzf/best.pt \
  --out outputs/runs/deepmimo_cnn_finetune

python scripts/evaluate_all.py \
  --dataset-type deepmimo \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --ckpt outputs/runs/deepmimo_cnn_finetune/best.pt \
  --config configs/deepmimo_cnn_finetune.yaml \
  --methods mrt zf rzf dft cnn \
  --out outputs/comparisons/deepmimo_cnn_finetune
```

Observed local DeepMIMO smoke details:

- Raw channel shape: `(131931, 1, 8, 1)`
- Converted project shape after grouping users and filtering zero-power groups: `(22178, 4, 8)`
- Filtered invalid group ratio: about `32.76%`
- Saved smoke tensor: `outputs/data/deepmimo_asu_campus_3p5_narrowband.pt`

DeepMIMO caveat:

- This is still a smoke-scale benchmark on a filtered narrowband tensor, not yet a full DeepMIMO study.
- The current learned smoke run slightly exceeds the filtered RZF baseline on this tensor, but that result should not be overstated as a full DeepMIMO benchmark.

## Sionna

Sionna is still a secondary extension path, not the current execution bottleneck. It remains useful for future differentiable end-to-end links.

Notes:

- Sionna PHY / SYS are PyTorch-based
- Sionna can be used later for end-to-end differentiable link studies
- Current project progress does not depend on Sionna being installed now
- The target environment should satisfy the current Sionna requirements from the official docs, including Python `3.11+` and modern PyTorch releases

The notebook scaffold remains:

- [notebooks/02_sionna_e2e_demo.ipynb](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/notebooks/02_sionna_e2e_demo.ipynb)

## Final Acceptance Commands

```bash
python -m compileall src scripts tests
pytest -q
python scripts/run_baselines.py --data outputs/data/synthetic_narrowband.pt --methods mrt zf rzf dft --out outputs/runs/baselines_synthetic
python scripts/pretrain.py --config configs/synthetic_cnn_pretrain.yaml --data outputs/data/synthetic_narrowband.pt --teacher rzf --out outputs/runs/cnn_pretrain_rzf
python scripts/train.py --config configs/synthetic_cnn_finetune.yaml --data outputs/data/synthetic_narrowband.pt --init-ckpt outputs/runs/cnn_pretrain_rzf/best.pt --out outputs/runs/cnn_finetune_rzf
python scripts/evaluate_all.py --data outputs/data/synthetic_narrowband.pt --ckpt outputs/runs/cnn_finetune_rzf/best.pt --config configs/synthetic_cnn_finetune.yaml --methods mrt zf rzf dft cnn --out outputs/comparisons/synthetic_cnn_finetune
python scripts/inspect_deepmimo.py --scenario asu_campus_3p5 --download --num-users 4 --narrowband
```
