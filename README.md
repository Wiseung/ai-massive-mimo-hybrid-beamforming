# AI Massive MIMO Hybrid Beamforming

Reproducible PyTorch-based single-GPU project for massive MIMO / mmWave beamforming and precoding. The current repository is centered on a verified synthetic CSI pipeline and fair benchmark comparison before pushing further into heavier DeepMIMO or Sionna features.

## Release Snapshot

### Synthetic Headline

| Method | Mean SE | Notes |
| --- | ---: | --- |
| RZF | 5.5771 | low-latency analytic reference |
| WMMSE | 5.8523 | strongest tested synthetic reference |
| WMMSE iter 5 | 5.8155 | reduced-iteration reference |
| Best unfolded WMMSE-lite | 5.8163 | `wmmse_iter_5` init, 3 layers, `distill=0.1`, `delta=1e-3` |
| CNN | 5.0524 | warm-started black-box baseline |

### DeepMIMO Headline

| Split | Best method | Mean SE | Scale |
| --- | --- | ---: | --- |
| random | WMMSE iter 5 | 1.0884 ± 0.0484 | K=4, Nt=8, Nsc=1 |
| contiguous | WMMSE iter 5 | 1.0664 ± 0.0000 | K=4, Nt=8, Nsc=1 |

### Latency Protocol

| Item | Value |
| --- | --- |
| device | CUDA when available |
| batch size | 512 |
| warmup runs | 20 |
| timed runs | 100 |
| include data transfer | false |

## Current Status

- Synthetic pipeline is verified end-to-end.
- Classical baseline results are verified.
- The original CNN training path ran successfully but its fair evaluation was weak.
- A fair unified evaluation path now exists through `scripts/evaluate_all.py`.
- Teacher warm-start and SNR-conditioned training were added.
- Warm-started CNN now approaches the RZF baseline on the synthetic benchmark.
- High-SNR-weighted fine-tuning did not materially improve the synthetic high-SNR gap.
- Mixed-teacher warm-start produced only a marginal synthetic gain over the RZF teacher.
- DeepMIMO dataset diagnostics and reproducible split generation were added.
- Fully-digital `fd_zf` / `fd_rzf` reference labels were added for unified comparison, but in the current digital-only MU-MISO setup they are reference aliases rather than stronger upper bounds.
- A residual refinement model around the RZF prior was added to target the remaining synthetic high-SNR gap.
- A small-scale MU-MISO WMMSE baseline is now implemented and evaluated on the synthetic benchmark.
- An unfolded-RZF refinement model is now implemented for structure-aware learned precoding.
- Cross-method latency is now standardized through `scripts/benchmark_latency.py` instead of mixing runtime numbers from different scripts.
- A previous `residual_wmmse` evaluation path had teacher leakage at inference; that path is now fixed and the fair `v2` result is documented instead of the leaked one.
- WMMSE distillation, `residual_wmmse`, and `unfolded_wmmse_lite` are implemented, but their fair results must now be interpreted under the unified latency protocol.
- DeepMIMO `v4` is installed locally and the `asu_campus_3p5` smoke path, baseline smoke benchmark, and a small learned smoke benchmark all ran successfully.
- DeepMIMO quick multi-seed benchmarking is implemented, but quick mode is explicitly not treated as a full benchmark.
- The current DeepMIMO quick `v2` benchmark still ran only `seed=1`, so its `std` entries remain `NaN` and it is not reported as a full multi-seed result.
- A contiguous DeepMIMO split benchmark is now available and should be interpreted as a harder location-generalization evaluation than the random split.
- A non-quick DeepMIMO full multi-seed benchmark has now run with `seeds=1,2,3`.
- A DeepMIMO random-split model-family benchmark across `seeds=1,2,3` is now available for `rzf`, `wmmse_iter_5`, `cnn`, `residual_rzf`, and `unfolded_wmmse_lite`.
- A contiguous DeepMIMO model-family benchmark across `seeds=1,2,3` is now available, together with an exported random-vs-contiguous comparison table and figure.
- A quick unfolded-WMMSE-lite ablation sweep is now exported and identifies the current best synthetic variant as `wmmse_iter_5` initialization with `3` refinement layers.
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

When `--methods` includes multiple learned models such as `cnn residual_rzf`, `scripts/evaluate_all.py` evaluates the checkpoint/config passed on the command line and auto-discovers the repository's default fair-evaluation artifacts for the other requested learned methods when those checkpoints already exist.

All exported gaps now use the same formula:

```text
gap_to_reference = (method_se - reference_se) / reference_se
```

## Latency Benchmark Protocol

Cross-method latency comparisons now come only from:

```bash
python scripts/benchmark_latency.py \
  --data outputs/data/synthetic_narrowband.pt \
  --methods mrt zf rzf dft wmmse wmmse_iter_5 cnn residual_rzf residual_wmmse unfolded_rzf unfolded_wmmse_lite \
  --batch-size 512 \
  --warmup-runs 20 \
  --timed-runs 100 \
  --out outputs/comparisons/latency
```

Default protocol:

- `device=auto` and CUDA when available
- `batch_size=512`
- `warmup_runs=20`
- `timed_runs=100`
- `include_data_transfer=false`

The unified latency artifacts are:

- `outputs/comparisons/latency/latency_table.csv`
- `outputs/comparisons/latency/latency_bar.png`

Older latency numbers from mixed evaluation scripts are not used anymore for model-family tables or Pareto plots.

Profiling of `unfolded_wmmse_lite` is now available through:

```bash
python scripts/benchmark_latency.py \
  --data outputs/data/synthetic_narrowband.pt \
  --methods mrt rzf wmmse_iter_1 wmmse_iter_2 wmmse_iter_5 wmmse cnn residual_rzf unfolded_rzf unfolded_wmmse_lite \
  --batch-size 512 \
  --warmup-runs 20 \
  --timed-runs 100 \
  --profile-method unfolded_wmmse_lite \
  --out outputs/comparisons/latency_v2
```

The current hotspot report is saved to:

- `outputs/comparisons/latency_v2/unfolded_wmmse_lite_profile.json`

On the current run, the dominant cost is the `wmmse_iter_2` initialization step, not the learnable refinement layers.

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

### Dataset Diagnostics

The current local DeepMIMO smoke tensor is summarized by:

- `outputs/reports/deepmimo_dataset_summary.json`
- `outputs/figures/deepmimo_channel_norm_hist.png`
- `outputs/figures/deepmimo_user_group_power.png`

Observed local diagnostics on `2026-05-10`:

- raw channel shape: `(131931, 1, 8, 1)`
- grouped project tensor shape after filtering: `(22178, 4, 8)`
- invalid user-group ratio before filtering: `32.76%`
- zero-power ratio after filtering: `0.0`
- low-power ratio after filtering: `0.198%`

User groups are currently formed by contiguous blocks of `K=4` users in the DeepMIMO receiver ordering after selecting one BS, then zero-power groups are removed before saving the project tensor.

### Reproducible Splits

DeepMIMO splits can now be generated explicitly:

```bash
python scripts/make_splits.py \
  --dataset-type deepmimo \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --split-mode random \
  --seed 42 \
  --out outputs/splits/deepmimo_asu_seed42.pt
```

Supported modes:

- `random`
- `contiguous`

`contiguous` uses the current user-group ordering and assigns the first `70%` to train, the middle `15%` to val, and the last `15%` to test.

### DeepMIMO Smoke Benchmark Status

The current DeepMIMO benchmark should still be described as a filtered smoke benchmark:

- the scenario is real and was downloaded/loaded locally
- the tensor conversion path is real
- baseline and learned smoke runs are real
- the current result set is not yet a full multi-scenario DeepMIMO conclusion

The existing learned smoke result on this filtered tensor showed a positive gap over filtered RZF:

- `mean_se = 0.760969`
- `mean_gap_to_rzf = +2.6913%`

That result is useful smoke evidence, but it must not be overstated as a complete DeepMIMO benchmark conclusion.

The current quick benchmark output is:

- `outputs/comparisons/deepmimo_multiseed_quick_v2/deepmimo_benchmark_summary.csv`
- `outputs/comparisons/deepmimo_multiseed_quick_v2/deepmimo_benchmark_summary.md`

It explicitly reports `num_seeds=1`.

The current full multi-seed benchmark output is:

- `outputs/comparisons/deepmimo_full_multiseed/deepmimo_benchmark_summary.csv`
- `outputs/comparisons/deepmimo_full_multiseed/deepmimo_benchmark_summary.md`

This run now reports `num_seeds=3` with:

- `mean_se = 0.7312 +- 0.0318`
- `mean_gap_to_rzf = +2.15% +- 3.96%`
- `mean_gap_to_strongest_reference = -0.65% +- 0.59%`

It is still a small-scale benchmark on the currently available tensor shape `K=4`, `Nt=8`, `Nsc=1`, not a final massive-array DeepMIMO study.

The current random-split DeepMIMO model-family benchmark output is:

- `outputs/comparisons/deepmimo_model_family_random/deepmimo_model_family_table.csv`
- `outputs/comparisons/deepmimo_model_family_random/deepmimo_model_family_mean_std.csv`

Across `seeds=1,2,3`, the current random-split summary is:

- `wmmse_iter_5`: `mean_se = 1.0884 +- 0.0484`, `latency = 76.03 ms`
- `unfolded_wmmse_lite`: `mean_se = 0.8606 +- 0.0426`, `latency = 35.25 ms`
- `residual_rzf`: `mean_se = 0.7188 +- 0.0359`, `latency = 1.24 ms`
- `rzf`: `mean_se = 0.7141 +- 0.0357`, `latency = 0.90 ms`
- `cnn`: `mean_se = 0.7114 +- 0.0148`, `latency = 0.80 ms`

This is still a filtered `K=4`, `Nt=8`, `Nsc=1` benchmark and should not be overstated as a large-scale DeepMIMO conclusion.

The current contiguous-split DeepMIMO model-family benchmark output is:

- `outputs/comparisons/deepmimo_model_family_contiguous/deepmimo_model_family_table.csv`
- `outputs/comparisons/deepmimo_model_family_contiguous/deepmimo_model_family_mean_std.csv`

Across `seeds=1,2,3`, the current contiguous-split summary is:

- `wmmse_iter_5`: `mean_se = 1.0664 +- 0.0000`, `latency = 45.76 ms`
- `unfolded_wmmse_lite`: `mean_se = 0.8719 +- 0.0000`, `latency = 23.95 ms`
- `rzf`: `mean_se = 0.7642 +- 0.0000`, `latency = 0.54 ms`
- `residual_rzf`: `mean_se = 0.7639 +- 0.0001`, `latency = 0.93 ms`
- `cnn`: `mean_se = 0.6575 +- 0.0276`, `latency = 0.63 ms`

The random-vs-contiguous comparison artifacts are:

- `outputs/comparisons/deepmimo_model_family_random_vs_contiguous.csv`
- `outputs/comparisons/deepmimo_model_family_random_vs_contiguous.md`
- `outputs/comparisons/deepmimo_model_family_random_vs_contiguous.png`

On this local filtered tensor, contiguous split does not uniformly reduce every method:

- `cnn` drops by about `-7.58%`
- `wmmse_iter_5` drops by about `-2.03%`
- `unfolded_wmmse_lite` rises by about `+1.31%`
- `rzf` rises by about `+7.02%`
- `residual_rzf` rises by about `+6.28%`

These are empirical outputs from the current `K=4`, `Nt=8`, `Nsc=1` tensor, not a general theorem about location generalization.

### DeepMIMO Benchmark Commands

Dataset analysis:

```bash
python scripts/analyze_deepmimo_dataset.py \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --out outputs/reports/deepmimo_dataset_summary
```

Residual-RZF smoke benchmark:

```bash
python scripts/train.py \
  --dataset-type deepmimo \
  --config configs/deepmimo_residual_rzf.yaml \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --split outputs/splits/deepmimo_asu_seed42.pt \
  --out outputs/runs/deepmimo_residual_rzf

python scripts/evaluate_all.py \
  --dataset-type deepmimo \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --split outputs/splits/deepmimo_asu_seed42.pt \
  --ckpt outputs/runs/deepmimo_residual_rzf/best.pt \
  --config configs/deepmimo_residual_rzf.yaml \
  --methods mrt zf rzf dft cnn residual_rzf \
  --out outputs/comparisons/deepmimo_residual_rzf
```

Quick multi-seed smoke benchmark:

```bash
python scripts/run_deepmimo_benchmark.py \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --seeds 1 2 3 \
  --quick \
  --out outputs/comparisons/deepmimo_multiseed_quick
```

`--quick` is intentionally a smoke/engineering mode. README and report treat it as such, not as a final scientific benchmark.

## Residual RZF Motivation

The remaining synthetic weakness is concentrated at `15/20 dB`, where direct CNN precoder prediction still trails RZF. The repository therefore adds a residual/refinement model that starts from an analytic `RZF` precoder and learns only a correction:

```text
F_pred = normalize_power(F_rzf + alpha * delta_F)
```

This structure is meant to preserve the strong communication prior at high SNR rather than asking the network to reconstruct a good precoder from scratch.

An additional unfolding-style model is now included:

```text
F_{t+1} = normalize_power(F_t + alpha_t * direction_t),  F_0 = RZF
```

The motivation is the same: a plain black-box CNN does not explicitly encode matrix-inversion and interference-cancellation structure, while residual and unfolded refinements do.

Current verified synthetic structured-model results:

- `residual_rzf`: `mean_se = 5.5771`, `mean_gap_to_rzf ~= 0`
- `unfolded_rzf`: `mean_se = 5.5858`, `mean_gap_to_rzf = +0.0816%`
- `unfolded_wmmse_lite`: `mean_se = 5.7729`, `mean_gap_to_rzf = +9.31%`, `mean_gap_to_wmmse = -1.50%`
- `residual_wmmse` after leakage fix: `mean_se = 5.5771`, `mean_gap_to_rzf ~= 0`, `mean_gap_to_wmmse = -9.45%`

`residual_wmmse` now means an `RZF`-prior residual model distilled toward `WMMSE` during training. It does not receive a WMMSE teacher at inference. Earlier teacher-assisted inference results are not treated as fair and are no longer used in the benchmark narrative.

This means `unfolded_rzf` slightly improves over `residual_rzf` and slightly exceeds `RZF`, but still remains below `WMMSE`. The stronger WMMSE-directed variant in fair synthetic evaluation is `unfolded_wmmse_lite`, not `residual_wmmse`.

## WMMSE Status

`WMMSE` is no longer a scaffold in the narrowband digital-only MU-MISO path. A small-scale implementation is now enabled in `run_baselines.py`.

Current verified synthetic WMMSE result is stronger than `RZF` over the tested SNR grid:

- `mean_se = 5.9024`
- stronger than `RZF` at `10/15/20 dB`

The WMMSE iteration sweep still gives a useful within-family convergence trend:

- `iter=5`: `mean_se = 5.8155`, `gap_to_full_wmmse = -0.66%`, `latency = 0.436 ms`
- `iter=50`: `mean_se = 5.8540`, `latency = 4.394 ms`

For cross-method comparisons, however, the repository now uses only the unified latency protocol in `outputs/comparisons/latency/latency_table.csv`. Under that protocol:

- `rzf`: `1.380 ms`
- `residual_wmmse`: `1.941 ms`
- `unfolded_rzf`: `3.791 ms`
- `unfolded_wmmse_lite`: `112.60 ms`
- `wmmse_iter_5`: `268.41 ms`
- `wmmse`: `2048.41 ms`

So the current SE-latency Pareto frontier is roughly:

- `mrt -> rzf -> unfolded_rzf -> unfolded_wmmse_lite -> wmmse_iter_5 -> wmmse`

Under the standardized protocol, the best current `unfolded_wmmse_lite` variant is no longer a low-latency point. It nearly matches `wmmse_iter_5` in SE, but it inherits much of the structured initialization cost and therefore remains slower than `wmmse_iter_5`.

Current `latency_v2` values for the user-requested method set are:

- `mrt = 0.234 ms`
- `rzf = 0.554 ms`
- `wmmse_iter_1 = 37.49 ms`
- `wmmse_iter_2 = 73.91 ms`
- `unfolded_wmmse_lite = 190.24 ms`
- `wmmse_iter_5 = 185.46 ms`
- `wmmse = 1146.11 ms`

The quick sweep shows:

- best SE variant: `wmmse_iter_5` init, `3` layers, `distill_weight=0.1`, `delta_norm_weight=1e-3`
- `mean_se = 5.8163`
- `gap_to_wmmse = -0.43%`
- `gap_to_wmmse_iter_5 = +0.0055%`
- unified latency-table inference = `190.24 ms`

This means the current best learned WMMSE-lite variant is best interpreted as an SE-matching structured approximation to `wmmse_iter_5`, not as a lower-latency replacement.

## Reproduction Commands

Synthetic:

```bash
python scripts/evaluate_all.py \
  --data outputs/data/synthetic_narrowband.pt \
  --ckpt outputs/runs/synthetic_unfolded_wmmse_lite_iter2/best.pt \
  --config configs/synthetic_unfolded_wmmse_lite_iter2.yaml \
  --methods mrt zf rzf dft wmmse wmmse_iter_5 unfolded_wmmse_lite \
  --out outputs/comparisons/synthetic_unfolded_wmmse_lite_iter2
```

DeepMIMO random:

```bash
python scripts/run_deepmimo_model_family_benchmark.py \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --seeds 1 2 3 \
  --split-mode random \
  --methods rzf wmmse_iter_5 cnn residual_rzf unfolded_wmmse_lite \
  --out outputs/comparisons/deepmimo_model_family_random
```

DeepMIMO contiguous:

```bash
python scripts/run_deepmimo_model_family_benchmark.py \
  --data outputs/data/deepmimo_asu_campus_3p5_narrowband.pt \
  --seeds 1 2 3 \
  --split-mode contiguous \
  --methods rzf wmmse_iter_5 cnn residual_rzf unfolded_wmmse_lite \
  --out outputs/comparisons/deepmimo_model_family_contiguous
```

Latency:

```bash
python scripts/benchmark_latency.py \
  --data outputs/data/synthetic_narrowband.pt \
  --methods mrt rzf wmmse_iter_1 wmmse_iter_2 wmmse_iter_5 wmmse cnn residual_rzf unfolded_rzf unfolded_wmmse_lite \
  --batch-size 512 \
  --warmup-runs 20 \
  --timed-runs 100 \
  --profile-method unfolded_wmmse_lite \
  --out outputs/comparisons/latency_v2
```

Sweep:

```bash
python scripts/sweep_unfolded_wmmse_lite.py \
  --data outputs/data/synthetic_narrowband.pt \
  --quick \
  --out outputs/comparisons/unfolded_wmmse_lite_sweep
```

## Known Limitations

- DeepMIMO current scale is only `K=4`, `Nt=8`, `Nsc=1`
- no wideband DeepMIMO result exists locally
- no Sionna end-to-end result exists locally
- current `unfolded_wmmse_lite` remains below `wmmse_iter_5` unless a better sweep variant is found

This implementation should still be treated as a practical narrowband benchmark for the current setup, not as a complete hybrid / wideband WMMSE study.

## Current Negative Results

The following findings are currently negative and are kept explicitly in the documentation:

- high-SNR loss weighting was largely ineffective
- mixed RZF/ZF teacher provided only a very small gain
- the synthetic high-SNR gap is not meaningfully improved by simple weighting or mixed teachers
- `residual_rzf` and `unfolded_rzf` still remain materially below `WMMSE`
- Sionna is not part of the current mainline acceptance path

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
