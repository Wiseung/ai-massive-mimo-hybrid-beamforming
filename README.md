# AI Massive MIMO Hybrid Beamforming

PyTorch/DeepMIMO benchmark for Massive MIMO beamforming with WMMSE, WMMSE-lite, and learned refinements.

Published releases:

- [`v0.1.0` benchmark prototype](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/docs/releases/v0.1.0.md)
- [`v0.2.0` optional Sionna PHY/OFDM demos](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/docs/releases/v0.2.0.md)

Current validated scope:

- Synthetic narrowband MU-MISO
- DeepMIMO `asu_campus_3p5` filtered benchmark
- `K=4`, `Nt=8`, `Nsc=1`
- no wideband yet
- no Sionna end-to-end benchmark yet

## Release Snapshot

### Synthetic Headline

| Method | Mean SE | Notes |
| --- | ---: | --- |
| RZF | 5.5771 | low-latency analytic reference |
| WMMSE | 5.8523 | strongest tested synthetic reference |
| WMMSE iter 5 | 5.8155 | reduced-iteration reference |
| Best unfolded WMMSE-lite | 5.8163 | `wmmse_iter_5` init, 3 layers, `distill=0.1`, `delta=1e-3` |
| CNN | 5.0524 | warm-started black-box baseline |

### Synthetic SE-Latency

| Method | Mean SE | Latency (ms) | Note |
| --- | ---: | ---: | --- |
| RZF | 5.5771 | 0.5538 | low-latency analytic baseline |
| WMMSE | 5.8523 | 1146.11 | strongest full reference |
| WMMSE iter 5 | 5.8155 | 185.46 | strong reduced-iteration reference |
| Best unfolded WMMSE-lite | 5.8163 | 190.24 | matches `wmmse_iter_5` in SE, slower in latency |
| CNN | 5.0524 | 2.0442 | learned black-box baseline |

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

### WMMSE-lite Sweep

- best config: `init_method=wmmse_iter_5`, `num_layers=3`, `distill_weight=0.1`, `delta_norm_weight=1e-3`
- `mean_se = 5.816346`
- `gap_to_wmmse = -0.4282%`
- `gap_to_wmmse_iter_5 = +0.0055%`

### Latency Hotspot

- `unfolded_wmmse_lite` hotspot is the WMMSE initializer
- profile: `init_computation_time_ms ~= 200.51`
- profile: `layer_refinement_time_ms ~= 3.29`
- learned refinement overhead is small; structured initialization dominates latency

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
- A contiguous DeepMIMO split benchmark is now available as an alternative location-ordered evaluation; on the current filtered tensor it should not be oversimplified as uniformly harder than the random split.
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
pip install sionna-no-rt
```

DeepMIMO notes:

- Prefer a Python `3.11+` environment for the DeepMIMO path.
- The DeepMIMO smoke test is independent from Sionna.
- Do not commit downloaded DeepMIMO scenarios or generated tensors into git.
- Do not commit raw DeepMIMO scenarios or large model checkpoints into git.

References:

- [Sionna documentation](https://nvlabs.github.io/sionna/index.html)
- [DeepMIMO documentation](https://www.deepmimo.net/docs/index.html)

## Experimental Sionna Branch

The Sionna demo path is an optional experimental branch and is not part of the `v0.1.0` benchmark acceptance path.

- Prefer a dedicated conda environment.
- Start with `pip install sionna-no-rt`.
- Treat the Sionna smoke scripts as coexistence checks with the current PyTorch beamforming project, not as a full Sionna end-to-end result.
- The current branch also includes Sionna PHY introspection, a minimal AWGN PHY smoke demo, and a small beamforming link demo.
- The current branch also includes OFDM API introspection, a ResourceGrid smoke demo, and an OFDM beamforming bridge demo.
- The current branch also includes a tiny differentiable beamformer smoke demo to verify backward/short-step optimization through an OFDM-style link.
- If a Sionna PHY component is unavailable or unstable for the current environment, the demo records an explicit torch fallback instead of silently pretending the path is fully Sionna-native.

## Optional Sionna PHY/OFDM Demos

Available on the published `v0.2.0` release and the earlier `feature/sionna-phy-ofdm-link` branch history.

- optional dependency: `sionna-no-rt`
- does not change `v0.1.0` benchmark claims
- no RT
- no ray tracing
- no 5G NR full stack
- toy differentiable beamformer only

```bash
python scripts/check_sionna_env.py
python scripts/inspect_sionna_api.py --out outputs/sionna_smoke/sionna_api_summary.json
python scripts/inspect_sionna_ofdm_api.py --out outputs/sionna_smoke/sionna_ofdm_api_summary.json
python scripts/sionna_phy_awgn_demo.py --out outputs/sionna_smoke/sionna_phy_awgn_summary.json
python scripts/sionna_ofdm_resource_grid_demo.py --out outputs/sionna_smoke/sionna_ofdm_resource_grid_summary.json
python scripts/sionna_ofdm_beamforming_bridge_demo.py --out outputs/sionna_smoke/sionna_ofdm_beamforming_bridge_summary.json
python scripts/check_differentiable_beamformer_gradients.py --out outputs/sionna_smoke/differentiable_beamformer_gradcheck.json
python scripts/sionna_ofdm_differentiable_beamforming_demo.py --out outputs/sionna_smoke/sionna_ofdm_differentiable_beamforming_summary.json
```

## Experimental Sionna OFDM Training

Available on `feature/sionna-learned-beamformer-training` as an optional post-`v0.2.0` experiment.

- optional dependency only: `sionna-no-rt`
- synthetic OFDM channel only
- multi-SNR link-level training
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- does not change `v0.1.0` or `v0.2.0` release claims

### Sionna OFDM Learned Training Status

Current mainline interpretation for this branch:

- `SionnaOFDMResidualRZFBeamformer` is the cleanest mainline.
- `SionnaOFDMResidualWMMSEDistilledBeamformer` is safe but gives only a tiny improvement.
- no learned model beats `WMMSE-iter5`.
- results remain synthetic OFDM only.

Compact headline table:

| Method | Status | mean_sum_rate | gap_to_rzf | gap_to_wmmse_iter_5 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| TinyNeuralBeamformer | full | `8.723585` | `-39.5834%` | `-39.9628%` | much weaker than analytic priors |
| Residual RZF | full | `17.657597` | `+0.0134%` | `-0.4999%` | current mainline |
| Unfolded Lite | full | `17.466006` | `-0.3550%` | `-0.8715%` | slower due to `wmmse_iter_2` initializer |
| Residual WMMSE-distill | full | `17.657607` | `+0.0135%` | `-0.4997%` | safe, near-null gain over residual RZF |
| RZF | full baseline | reference | `0%` | about `-0.513%` | fast analytic baseline |
| WMMSE-iter5 | full baseline | reference | above RZF | `0%` | strongest reduced-iteration baseline |

Quick-only results:

- multi-seed robustness is `--quick`, not a full exhaustive benchmark
- scale sweep is `--quick`
- train-SNR ablation is `--quick`
- distillation-weight sweep is `--quick`

Current learned OFDM training status:

- `TinyNeuralBeamformer` full result:
  `mean_sum_rate = 8.723585`, `gap_to_rzf = -39.5834%`, `gap_to_wmmse_iter_5 = -39.9628%`
- `SionnaOFDMResidualRZFBeamformer` full result:
  `mean_sum_rate = 17.657597`, `gap_to_rzf = +0.0134%`, `gap_to_wmmse_iter_5 = -0.4999%`
- `SionnaOFDMUnfoldedLiteBeamformer` full result:
  `mean_sum_rate = 17.466006`, `gap_to_rzf = -0.3550%`, `gap_to_wmmse_iter_5 = -0.8715%`
- `SionnaOFDMResidualWMMSEDistilledBeamformer` full result:
  `mean_sum_rate = 17.657607`, `gap_to_rzf = +0.0135%`, `gap_to_wmmse_iter_5 = -0.4997%`
- high-SNR gap is dramatically improved by communication priors relative to `TinyNeuralBeamformer`
- `SionnaOFDMResidualWMMSEDistilledBeamformer` uses `WMMSE-iter5` only as a training-time teacher target; inference inputs remain `H_f + F_rzf + snr`
- teacher leakage audit reports:
  `teacher_used_during_training = true`,
  `teacher_used_during_inference = false`,
  `model_forward_calls_wmmse = false`,
  `leakage_detected = false`
- distillation weight quick sweep (`0.0, 0.05, 0.1, 0.5, 1.0`) shows almost no practical separation in gap to `WMMSE-iter5`; no weight produced a meaningful jump beyond the residual-RZF operating point
- current recommended next-stage mainline for this branch remains `SionnaOFDMResidualRZFBeamformer` for clarity, while `SionnaOFDMResidualWMMSEDistilledBeamformer` is kept as a documented near-null result
- quick multi-seed robustness (`seeds = 1,2,3`) keeps `SionnaOFDMResidualRZFBeamformer` as the strongest learned method:
  `mean_sum_rate_mean = 17.656879 +/- 0.020110`,
  `gap_to_rzf_mean = +0.0259% +/- 0.2038%`,
  `gap_to_wmmse_iter_5_mean = -0.5732% +/- 0.1216%`
- OFDM inference latency benchmark (`B=128, Nsc=8, K=4, Nt=16`) shows:
  `rzf = 1.069 ms`, `tiny = 0.848 ms`, `residual_rzf = 2.246 ms`,
  `unfolded_lite = 56.120 ms`, `wmmse_iter_2 = 58.532 ms`, `wmmse_iter_5 = 192.120 ms`
- current residual correction analysis indicates a small refinement around RZF rather than a robust RZF-beating gain:
  `mean_delta_norm_ratio = 0.052996`, `mean_relative_se_gain_over_rzf = +0.004854%`, `alpha ~= 0.0979`
- quick train-SNR ablation does not show a meaningful sensitivity yet:
  `high_only [15, 20]` is marginally best, but the mean-gap span to RZF is only about `2.9e-5`
- quick scale sweep keeps residual-RZF effectively on the RZF operating point and below `WMMSE-iter5` by about `-0.3992%` on average across evaluated settings

```bash
python scripts/train_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_learned_beamformer.yaml \
  --out outputs/runs/sionna_ofdm_learned_beamformer_smoke \
  --smoke

python scripts/evaluate_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_learned_beamformer.yaml \
  --ckpt outputs/runs/sionna_ofdm_learned_beamformer_smoke/best.pt \
  --out outputs/comparisons/sionna_ofdm_learned_beamformer_smoke

python scripts/train_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_residual_rzf.yaml \
  --out outputs/runs/sionna_ofdm_residual_rzf_smoke \
  --smoke

python scripts/evaluate_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_residual_rzf.yaml \
  --ckpt outputs/runs/sionna_ofdm_residual_rzf_smoke/best.pt \
  --out outputs/comparisons/sionna_ofdm_residual_rzf_smoke

python scripts/train_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_unfolded_lite.yaml \
  --out outputs/runs/sionna_ofdm_unfolded_lite_smoke \
  --smoke

python scripts/evaluate_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_unfolded_lite.yaml \
  --ckpt outputs/runs/sionna_ofdm_unfolded_lite_smoke/best.pt \
  --out outputs/comparisons/sionna_ofdm_unfolded_lite_smoke

python scripts/compare_sionna_ofdm_training_runs.py \
  --tiny outputs/comparisons/sionna_ofdm_learned_beamformer \
  --residual outputs/comparisons/sionna_ofdm_residual_rzf \
  --unfolded outputs/comparisons/sionna_ofdm_unfolded_lite \
  --out outputs/comparisons/sionna_ofdm_training_family

python scripts/run_sionna_ofdm_multiseed_benchmark.py \
  --configs configs/sionna_ofdm_learned_beamformer.yaml \
            configs/sionna_ofdm_residual_rzf.yaml \
            configs/sionna_ofdm_unfolded_lite.yaml \
  --seeds 1 2 3 \
  --out outputs/comparisons/sionna_ofdm_multiseed \
  --quick

python scripts/benchmark_sionna_ofdm_models.py \
  --out outputs/comparisons/sionna_ofdm_latency

python scripts/analyze_sionna_residual_corrections.py \
  --config configs/sionna_ofdm_residual_rzf.yaml \
  --ckpt outputs/runs/sionna_ofdm_residual_rzf/best.pt \
  --out outputs/comparisons/sionna_ofdm_residual_analysis

python scripts/sweep_sionna_ofdm_scale.py \
  --quick \
  --out outputs/comparisons/sionna_ofdm_scale_sweep

python scripts/sweep_sionna_train_snr.py \
  --model sionna_ofdm_residual_rzf \
  --quick \
  --out outputs/comparisons/sionna_ofdm_snr_ablation

python scripts/train_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_residual_wmmse_distill.yaml \
  --out outputs/runs/sionna_ofdm_residual_wmmse_distill_smoke \
  --smoke

python scripts/evaluate_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_residual_wmmse_distill.yaml \
  --ckpt outputs/runs/sionna_ofdm_residual_wmmse_distill_smoke/best.pt \
  --out outputs/comparisons/sionna_ofdm_residual_wmmse_distill_smoke

python scripts/train_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_residual_wmmse_distill.yaml \
  --out outputs/runs/sionna_ofdm_residual_wmmse_distill

python scripts/evaluate_sionna_ofdm_beamformer.py \
  --config configs/sionna_ofdm_residual_wmmse_distill.yaml \
  --ckpt outputs/runs/sionna_ofdm_residual_wmmse_distill/best.pt \
  --out outputs/comparisons/sionna_ofdm_residual_wmmse_distill

python scripts/sweep_sionna_wmmse_distill_weight.py \
  --config configs/sionna_ofdm_residual_wmmse_distill.yaml \
  --weights 0.0 0.05 0.1 0.5 1.0 \
  --quick \
  --out outputs/comparisons/sionna_ofdm_wmmse_distill_sweep

python scripts/audit_sionna_teacher_leakage.py \
  --config configs/sionna_ofdm_residual_wmmse_distill.yaml \
  --ckpt outputs/runs/sionna_ofdm_residual_wmmse_distill/best.pt \
  --out outputs/comparisons/sionna_ofdm_teacher_leakage_audit

python scripts/compare_sionna_ofdm_training_runs.py \
  --tiny outputs/comparisons/sionna_ofdm_learned_beamformer \
  --residual outputs/comparisons/sionna_ofdm_residual_rzf \
  --unfolded outputs/comparisons/sionna_ofdm_unfolded_lite \
  --wmmse-distill outputs/comparisons/sionna_ofdm_residual_wmmse_distill \
  --out outputs/comparisons/sionna_ofdm_training_family_v2
```

See [`docs/sionna_learned_beamformer_training.md`](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/docs/sionna_learned_beamformer_training.md) for the experimental training scope and limitations.

## Experimental Sionna-Native OFDM Link Chain

Available on `feature/sionna-native-ofdm-link-chain` only.

- optional dependency only: `sionna-no-rt`
- feature-branch-only integration work
- does not change `v0.3.0` claims
- current status: `v0.4.0` candidate only
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- native PHY/OFDM chain exploration only, not production e2e

Reproduction:

```bash
python scripts/check_sionna_env.py
python scripts/audit_sionna_native_ofdm_components.py \
  --out outputs/sionna_native_chain/ofdm_component_audit.json
python scripts/sionna_native_ofdm_baseline_chain.py \
  --out outputs/sionna_native_chain/baseline_chain_summary.json
python scripts/audit_sionna_precoding_components.py \
  --out outputs/sionna_native_chain/precoding_component_audit.json
python scripts/audit_sionna_resource_grid_pilots.py \
  --out outputs/sionna_native_chain/pilot_pattern_audit.json
python scripts/sionna_native_estimator_equalizer_demo.py \
  --out outputs/sionna_native_chain/estimator_equalizer_demo_summary.json
python scripts/sionna_native_ofdm_beamforming_chain.py \
  --out outputs/sionna_native_chain/beamforming_chain_summary.json
python scripts/sionna_native_ofdm_beamforming_chain.py \
  --out outputs/sionna_native_chain/beamforming_receiver_chain_v2_summary.json \
  --enable-receiver-chain \
  --receiver-mode auto \
  --trace-shapes
python scripts/sionna_native_ofdm_learned_beamforming_chain.py \
  --out outputs/sionna_native_chain/learned_beamforming_receiver_summary.json \
  --receiver-mode auto \
  --trace-shapes
python scripts/compare_sionna_native_chains.py \
  --baseline outputs/sionna_native_chain/baseline_chain_summary.json \
  --beamforming outputs/sionna_native_chain/beamforming_chain_summary.json \
  --receiver outputs/sionna_native_chain/beamforming_receiver_chain_v2_summary.json \
  --metrics outputs/sionna_native_chain/beamforming_receiver_chain_v2_metrics.csv \
  --out outputs/sionna_native_chain
python scripts/compare_sionna_native_learned_beamforming.py \
  --analytic-summary outputs/sionna_native_chain/beamforming_receiver_chain_v2_summary.json \
  --analytic-metrics outputs/sionna_native_chain/beamforming_receiver_chain_v2_metrics.csv \
  --learned-summary outputs/sionna_native_chain/learned_beamforming_receiver_summary.json \
  --learned-metrics outputs/sionna_native_chain/learned_beamforming_receiver_metrics.csv \
  --out outputs/sionna_native_chain
python scripts/run_sionna_native_learned_chain_minibench.py \
  --out outputs/sionna_native_chain/native_learned_minibench
python scripts/generate_sionna_native_chain_artifact_manifest.py \
  --out outputs/sionna_native_chain/native_chain_artifact_manifest.json
python scripts/reproduce_sionna_native_chain_minimal.py \
  --out outputs/repro/sionna_native_chain_minimal_summary.json
```

See [`docs/sionna_native_ofdm_link_chain.md`](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/docs/sionna_native_ofdm_link_chain.md) for the intended beamforming insertion point and current chain limitations.

Current post-`v0.4.0` next-step focus:

- reduce the current `project-H_f-assisted` limitation
- audit whether real Sionna channel tensors can be converted into project `H_f=(B,Nsc,K,Nt)`
- keep the existing native receiver path intact while testing a more native channel bridge
- do not reinterpret this work as a full native-only benchmark unless channel, precoder, and receiver paths are all consistently native

Published status: `v0.5.0` for the optional Sionna-native channel-extraction bridge.
Published status: `v0.6.0` for the provenance-aware CSI interface on top of that bridge.
Current branch status: `v0.7.0` candidate for CSI consumer unification across analytic, learned, native-chain, and comparison paths.
Current branch next step: `feature/sionna-native-precoder-interface-bridge` for a standardized `PrecoderOutput` bridge on top of `ExtractedCSI`.

Current channel-extraction branch result:

- `OFDMChannel(return_channel=True)` can be converted into project `H_f=(B,Nsc,K,Nt)` under the current MU downlink assumptions
- the current extraction demo succeeds with `sionna_channel_tensor_shape=[8,4,1,1,16,2,19]` and `extracted_h_f_shape=[8,16,4,16]`
- the native-channel-assisted beamforming demo succeeds with `project_h_f_assisted=false`
- this reduces the old project-assisted limitation, but it still does not justify a full native-only benchmark claim
- axis validation now spot-checks the bridge against the raw Sionna tensor and passes with `spot_check_max_abs_diff=0.0`
- the current pilot-aware bridge uses OFDM symbol `1` as the first data-bearing symbol because OFDM symbol `0` is reserved for pilots
- extracted-H quick consistency benchmarking (`seeds=1,2,3`, `snr=0,5,10,15,20 dB`, quick mode) keeps `extraction_success=true` and `native_receiver_success=true`, but proxy/native exact rank agreement is only `0.226667`
- therefore project-side proxy metrics should not be treated as a reliable substitute for the native receiver metric under the extracted-H path
- quick extracted-H consistency currently shows `learned_residual_rzf` below `project_rzf` on average (`-1.681560%`) and above `project_wmmse_iter_5` on average (`+4.735929%`) in this limited quick setting only; this is not enough to generalize a stable learned-`>`-`WMMSE-iter5` claim
- extraction-config sweep keeps successful extraction across `first_data`, `last_data`, `all_data_average`, `all_effective`, `center_8`, `center_16`, and optional normalization, with the current default still `first_data + all_effective + normalize=false`
- compared with the earlier project-assisted native-chain path, the extracted-H path changes the single-run method ranking and shrinks the project-assisted limitation, but it still remains native-channel-assisted rather than full native-only

See [`docs/sionna_native_channel_extraction.md`](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/docs/sionna_native_channel_extraction.md) for the extraction path, shape mapping, and current limits.

Compact extracted-H result table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| channel tensor shape | `[8,4,1,1,16,2,19]` | observed `h=[B,rx,rx_ant,tx,tx_ant,ofdm_symbol,fft_bin]` |
| extracted `H_f` shape | `[8,16,4,16]` | compatible with project `H_f=(B,Nsc,K,Nt)` interface |
| axis validation | `spot_check_max_abs_diff=0.0` | current axis mapping is consistent |
| native receiver success | `true` | current path is native-channel-assisted + native-receiver-assisted |
| proxy/native rank agreement | `0.226667` | proxy metric cannot replace native receiver metric |

Current supported wording:

- extracted-H_f path reduces the `project-H_f-assisted` limitation
- current supported description is `native-channel-assisted + native-receiver-assisted`
- not full native-only benchmark
- proxy metric cannot replace native receiver metric
- no RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

Current post-`v0.5.0` branch focus:

- standardize extracted `H_f` into a reusable `ExtractedCSI` interface instead of open-coded transpose/squeeze bridges
- keep `H_f` normalized to project shape `B,Nsc,K,Nt`
- attach provenance metadata such as original Sionna channel shape, original axes, selected data symbol, and effective subcarrier indices
- keep the current boundary as native-channel-assisted plus native-receiver-assisted, not full native-only

Current CSI-interface branch result:

- `ExtractedCSI` now records `source`, `source_component`, `axes`, `shape`, `project_h_f_assisted`, `extracted_h_f_used`, `full_native_only`, and nested provenance metadata
- CSI audit passes with `h_f_shape_ok=true`, `axes_metadata_complete=true`, `selected_data_symbol_not_pilot=true`, `project_h_f_assisted=false`, and `full_native_only=false`
- the CSI-backed beamforming chain succeeds for `project_rzf`, `project_wmmse_iter_5`, `learned_residual_rzf`, and `learned_residual_wmmse_distill`
- learned CSI-backed runs keep `teacher_used_during_inference=false`
- the same-batch equivalence validation now passes with `same_channel_tensor_used=true`, `same_bits_used=true`, `same_noise_config_used=true`, `same_receiver_config_used=true`, `numeric_consistency_within_tolerance=true`, and `ranking_consistent=true`
- under a shared realization, raw extracted-H and CSI-backed paths are numerically consistent for the current evaluated methods; this is the correct place to make an equivalence claim
- the earlier raw extracted-H vs CSI-backed mismatch is now audited as `cross_run_comparison_without_shared_realization`, not as evidence of a CSI-interface bug
- the cross-run comparison still introduces no extra fallback, but it must be read as provenance/schema comparison rather than a strict equivalence test

Compact CSI result table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| CSI audit | `passed` | provenance metadata is complete enough for current project and learned consumers |
| CSI-backed beamforming | `native_receiver_success=true` | CSI object enters native-channel-assisted + native-receiver-assisted path |
| same-batch equivalence | `passed` | raw extracted-H and CSI-backed paths are numerically consistent under shared realization |
| previous mismatch root cause | `cross_run_comparison_without_shared_realization` | earlier mismatch was cross-run, not CSI-interface bug evidence |

Current CSI consumer-unification status:

- `ExtractedCSI` is now the preferred input interface for key analytic and learned consumers
- `compute_project_precoder_per_subcarrier(...)` accepts `ExtractedCSI`, raw `H_f`, and dict inputs with `h_f`
- `infer_learned_precoder(...)` accepts `ExtractedCSI` directly and records CSI provenance in inference metadata
- the unified CSI consumer demo reuses one shared `ExtractedCSI` object across analytic, learned, and native-receiver consumers
- raw `H_f` remains as a backward-compatible fallback for older scripts and tests
- this improves provenance clarity and consumer consistency, but still does not make the system full native-only

Compact CSI consumer-unification table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| total consumers audited | `15` | coverage now includes analytic, learned, native-chain, comparison, tests, and docs paths |
| raw-only high-priority paths | `0` | no key consumer remains blocked on raw-only `H_f` |
| already support both | `12` | most important consumers accept `ExtractedCSI` and keep raw fallback |
| all consumers accept CSI | `true` | unified demo runs all current key methods from one shared CSI object |
| no new fallback introduced | `true` | CSI consumer unification did not add new fallback behavior |
| unified vs baseline strict equivalence claim allowed | `false` | the current unified-vs-baseline artifact is cross-run only |
| analytic consumers | `ExtractedCSI + raw fallback` | project `RZF/WMMSE` interfaces now accept standardized CSI |
| learned consumers | `ExtractedCSI + raw fallback` | learned inference accepts CSI directly with `teacher_used_during_inference=false` |
| unified CSI demo | `same CSI object reused` | one `ExtractedCSI` drives analytic, learned, and native receiver paths |
| unified vs baseline comparison | `cross-run comparison only` | confirms no new fallback and consistent CSI acceptance, but should not be read as a same-batch equivalence claim |

Current `v0.7.0` candidate interpretation:

- `ExtractedCSI` is the preferred input interface for current key consumers
- raw `H_f` remains a backward-compatible fallback
- `raw_only_high_priority_paths = 0`
- unified consumer demo reports `all_consumers_accept_csi = true`
- unified-vs-baseline remains `comparison_type = cross_run_comparison`
- `strict_equivalence_claim_allowed = false` for the unified-vs-baseline artifact
- strict same-batch equivalence remains the `v0.6.0` validation path, not the `v0.7.0` cross-run comparison
- not full native-only benchmark
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

Current post-`v0.7.0` / `v0.8.0` candidate branch focus:

- standardize precoder / beamformer outputs as a reusable `PrecoderOutput` object instead of passing raw `F_f=(B,Nsc,Nt,K)` tensors everywhere
- keep analytic and learned methods compatible with the current project-side bridge while making the native receiver path consume one auditable precoder container
- track `teacher_used_during_inference=false`, power normalization, checkpoint provenance, and project-vs-native precoder boundary directly on the output object
- keep raw `F_f` as a backward-compatible fallback for older scripts and targeted cross-run comparisons
- keep the current boundary as native-channel-assisted plus native-receiver-assisted, not full native-only

Current PrecoderOutput bridge status:

- `PrecoderOutput` now records `source`, `method`, `input_csi_summary`, `axes`, `shape`, `power_normalized`, `power_norm`, `teacher_used_during_inference`, `project_side_precoder`, `sionna_native_precoder`, `full_native_only`, and nested provenance metadata
- analytic `project_rzf` / `project_wmmse_iter_5` can emit `PrecoderOutput` through `return_precoder_output=True`
- learned `learned_residual_rzf` / `learned_residual_wmmse_distill` can emit `PrecoderOutput` through `return_precoder_output=True`
- `teacher_used_during_inference=false` is tracked directly in the learned `PrecoderOutput` summary
- the native receiver bridge now accepts either `PrecoderOutput` or raw `F_f`, with `PrecoderOutput` as the preferred interface for the unified demo path
- `ExtractedCSI` is the preferred input interface and `PrecoderOutput` is the preferred output interface for the current mainline Sionna-assisted path
- raw `H_f` and raw `F_f` remain backward-compatible fallbacks
- raw `F_f` remains a backward-compatible fallback
- same-batch raw-`F_f` vs `PrecoderOutput` validation now passes under one shared CSI / `F_f` / bits / noise / receiver-config realization
- the earlier raw-`F_f` vs `PrecoderOutput` ranking mismatch is now explicitly treated as a cross-run comparison artifact, not direct `PrecoderOutput` bug evidence

Compact PrecoderOutput table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| PrecoderOutput schema | `implemented` | standardized `F_f=(B,Nsc,Nt,K)` output container with provenance |
| analytic emit PrecoderOutput | `supported` | project `RZF/WMMSE` can return container or raw fallback |
| learned emit PrecoderOutput | `supported` | learned residual methods can return container or raw fallback |
| native receiver accepts PrecoderOutput | `supported` | receiver bridge consumes standardized output object |
| raw-only high-priority precoder gaps | `0` | no key high-priority path must stay raw-only |
| same-batch raw-vs-PrecoderOutput equivalence | `passed` | shared-realization validation gives `max_abs_diff_sum_rate=0.0`, `max_abs_diff_symbol_mse=0.0`, `max_abs_diff_sinr_db=0.0` |
| previous raw-vs-PrecoderOutput mismatch root cause | `cross_run_comparison_without_shared_csi_and_precoder_realization` | prior ranking mismatch was caused by comparing independent reruns |
| strict raw-vs-PrecoderOutput equivalence claim on cross-run artifact | `false` | only same-batch validation can support the strict numerical-consistency claim |
| current v0.8.0 candidate status | `release hardening` | manifest, minimal reproduction, release notes, and PR text prepared around the interface bridge |

Current PrecoderOutput interpretation:

- `PrecoderOutput` is numerically consistent under one shared CSI / `F_f` realization
- the old raw-vs-PrecoderOutput comparison remains a cross-run comparison, not a strict equivalence test
- no new fallback is introduced by the `PrecoderOutput` bridge
- this still does not make the system full native-only
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

Current CSI-interface validation commands:

```bash
python scripts/audit_sionna_csi_interface.py \
  --out outputs/sionna_channel_extraction/csi_interface_audit.json
python scripts/sionna_csi_backed_beamforming_chain.py \
  --out outputs/sionna_channel_extraction/csi_backed_beamforming_summary.json \
  --receiver-mode auto \
  --seed 0
python scripts/validate_csi_same_batch_equivalence.py \
  --out outputs/sionna_channel_extraction/csi_same_batch_equivalence.json
python scripts/audit_csi_raw_comparison_mismatch.py \
  --raw outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv \
  --csi outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --out outputs/sionna_channel_extraction
python scripts/compare_csi_backed_vs_raw_extracted_h.py \
  --raw outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv \
  --csi outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --out outputs/sionna_channel_extraction
python scripts/generate_sionna_csi_interface_artifact_manifest.py \
  --out outputs/sionna_channel_extraction/csi_interface_artifact_manifest.json
python scripts/reproduce_sionna_csi_interface_minimal.py \
  --out outputs/repro/sionna_csi_interface_minimal_summary.json
python scripts/audit_csi_consumers.py \
  --out outputs/sionna_channel_extraction/csi_consumer_audit.json
python scripts/demo_unified_csi_consumers.py \
  --out outputs/sionna_channel_extraction/unified_csi_consumers_summary.json
python scripts/compare_unified_csi_consumers.py \
  --baseline outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --unified outputs/sionna_channel_extraction/unified_csi_consumers_metrics.csv \
  --out outputs/sionna_channel_extraction
python scripts/generate_sionna_csi_consumer_artifact_manifest.py \
  --out outputs/sionna_channel_extraction/csi_consumer_artifact_manifest.json
python scripts/reproduce_sionna_csi_consumer_minimal.py \
  --out outputs/repro/sionna_csi_consumer_minimal_summary.json
python scripts/validate_precoder_output_same_batch_equivalence.py \
  --out outputs/sionna_channel_extraction/precoder_output_same_batch_equivalence.json
python scripts/audit_precoder_output_comparison_mismatch.py \
  --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv \
  --out outputs/sionna_channel_extraction
python scripts/compare_raw_ff_vs_precoder_output.py \
  --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv \
  --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv \
  --out outputs/sionna_channel_extraction
python scripts/generate_sionna_precoder_interface_artifact_manifest.py \
  --out outputs/sionna_channel_extraction/precoder_interface_artifact_manifest.json
python scripts/reproduce_sionna_precoder_interface_minimal.py \
  --out outputs/repro/sionna_precoder_interface_minimal_summary.json
```

Current channel-extraction validation commands:

```bash
python scripts/validate_sionna_extracted_hf_axes.py \
  --out outputs/sionna_channel_extraction/hf_axis_validation.json
python scripts/benchmark_sionna_extracted_h_consistency.py \
  --out outputs/sionna_channel_extraction/extracted_h_consistency \
  --seeds 1 2 3 \
  --snrs 0 5 10 15 20 \
  --quick
python scripts/sweep_sionna_channel_extraction_config.py \
  --quick \
  --out outputs/sionna_channel_extraction/extraction_config_sweep
python scripts/compare_project_hf_vs_extracted_hf.py \
  --project outputs/sionna_native_chain/learned_beamforming_receiver_metrics.csv \
  --extracted outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv \
  --consistency outputs/sionna_channel_extraction/extracted_h_consistency/metrics.csv \
  --out outputs/sionna_channel_extraction/project_vs_extracted_hf
```

Current branch status:

- `RZFPrecoder` is available in Sionna 2.0.1, but its expected tensor layout is not the same as the repository's `H_f=(B,Nsc,K,Nt)` project-side precoder path
- current clean mainline remains project frequency-domain precoder insertion
- current native-precoder probe branch adds an adapter-focused audit path for `ExtractedCSI -> Sionna RZFPrecoder input -> PrecoderOutput`
- current native-precoder interpretation remains compatibility mapping only: direct replacement is not yet the supported mainline
- current probe shows `RZFPrecoder` is callable, can be converted into `PrecoderOutput`, and can enter the current native receiver path through the adapter bridge
- `project_rzf` and `project_wmmse_iter_5` both improve strongly over `no_precoding` in the current beamforming-chain proxy metrics
- pilot-pattern audit shows that `LSChannelEstimator` requires a non-empty pilot pattern; `pilot_pattern=\"kronecker\"` with `pilot_ofdm_symbol_indices=[0]` is the current minimal working config
- the minimal estimator/equalizer demo succeeds with a real Sionna pilot-based receiver chain
- the shape-trace audit shows the earlier beamformed failure `shape '[16,1,1,0]'` came from a pilot-only grid with `num_data_symbols=0`, not from a random receiver bug
- the StreamManagement audit shows the beamformed downlink should use `num_tx=1`, `num_streams_per_tx=K`, and `rx_tx_association=ones(K,1)`
- the beamformed receiver chain now supports explicit `receiver-mode {proxy,native,auto}`:
  `proxy` keeps project-side metrics only,
  `native` requires the real Sionna receiver path,
  `auto` tries native first and records exact fallback stage/reason if needed
- the current `receiver-mode=auto` validation succeeds for `no_precoding`, `project_rzf`, `project_wmmse_iter_2`, and `project_wmmse_iter_5` through a real Sionna receiver path
- learned insertion into the same native receiver path now succeeds for `learned_residual_rzf` and `learned_residual_wmmse_distill`
- both learned native-chain methods keep `teacher_used_during_inference=false`
- current native-chain learned comparison is still synthetic and project-H_f-assisted; it is not a full native-only benchmark
- both learned methods enter a real Sionna receiver path while still consuming project-assisted `H_f` / precoder inputs
- fallback proxy metrics are still not described as full Sionna-native receiver results when native receiver mode is not used
- this branch therefore should still be read as an integration experiment, not a production e2e chain

Current native precoder API probe status:

- `sionna.phy.ofdm.RZFPrecoder` is callable on the current Sionna 2.0.1 install
- its native contract remains resource-grid-centric and higher rank than project `H_f=(B,Nsc,K,Nt)`
- the current adapter path can map one `ExtractedCSI` object into a minimal native RZF probe
- the current adapter path can convert the native RZF output back into `PrecoderOutput`
- same-realization validation now shows the converted native output is receiver-compatible but still `close_but_different` from `project_rzf`, not strictly equivalent
- quick `seed={1,2,3}` / `snr={0,5,10,15,20}` sweep keeps `RZFPrecoder` callable, convertible, and receiver-compatible on all evaluated rows
- `sionna_rzf_precoder` can now be included as an optional native method in the unified CSI + PrecoderOutput demo path
- `sionna_native_precoder=true` is acceptable for the adapter-produced native method output, but `project_rzf` strict equivalence is still `false`
- current `v0.9.0` candidate focus is release hardening around the optional native-method bridge, not project-side precoder replacement
- no Sionna RT
- no ray tracing
- no 5G NR full stack
- optional dependency only

Compact native precoder table:

| Item | Current result | Interpretation |
| --- | --- | --- |
| `sionna_rzf_available` | `true` | installed Sionna exposes `RZFPrecoder` |
| `sionna_rzf_callable` | `true` | minimal native call path works on the current environment |
| `converted_to_precoder_output` | `true` | native output can be mapped into project `F_f=(B,Nsc,Nt,K)` |
| `native_receiver_success` | `true` | converted native output enters the current native receiver path |
| `relationship_status` | `close_but_different` | same-realization semantics align, but not strict equivalence |
| `strict_equivalence_claim_allowed` | `false` | do not label `project_rzf` and Sionna RZF as strictly equivalent |
| `full_native_only` | `false` | still not a full native-only benchmark |

Current `v0.4.0` candidate comparison at the validated native insertion point:

| Method | Native receiver success | Teacher inference | Approx sum-rate | Gap vs `project_rzf` | Gap vs `project_wmmse_iter_5` |
| --- | --- | --- | ---: | ---: | ---: |
| `project_rzf` | `true` | `false` | `18.689259` | `0.000000%` | `+2.122005%` |
| `project_wmmse_iter_5` | `true` | `false` | `18.300722` | `-2.077917%` | `0.000000%` |
| `learned_residual_rzf` | `true` | `false` | `18.250900` | `-2.345509%` | `-0.272240%` |
| `learned_residual_wmmse_distill` | `true` | `false` | `18.381174` | `-1.648458%` | `+0.439611%` |

Interpretation:

- `learned_residual_rzf` remains the clean mainline learned insertion for the native chain
- `learned_residual_wmmse_distill` is a valid secondary variant and is slightly stronger than `learned_residual_rzf` in the current one-shot native-chain run
- neither result should be generalized into a stable, universal claim that learned methods beat `WMMSE-iter5`

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

The artifact manifest produced by `scripts/generate_artifact_manifest.py` is a result index. It is not a dataset archive and does not imply that raw DeepMIMO data or large checkpoints are tracked in git.

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

Quick smoke:

```bash
python scripts/reproduce_minimal.py \
  --out outputs/repro/minimal_repro_summary.json
```

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

Artifact manifest:

```bash
python scripts/generate_artifact_manifest.py \
  --out outputs/artifact_manifest.json
```

## Known Limitations

- DeepMIMO current scale is only `K=4`, `Nt=8`, `Nsc=1`
- no wideband DeepMIMO result exists locally
- no Sionna end-to-end result exists locally
- best `unfolded_wmmse_lite` matches `wmmse_iter_5` in SE but not in latency
- `unfolded_wmmse_lite` currently depends on a WMMSE initializer
- hybrid analog / RF constrained training is not the primary validated release path

This implementation should still be treated as a practical narrowband benchmark for the current setup, not as a complete hybrid / wideband WMMSE study.

## Current Negative Results

The following findings are currently negative and are kept explicitly in the documentation:

- high-SNR loss weighting was largely ineffective
- mixed RZF/ZF teacher provided only a very small gain
- the synthetic high-SNR gap is not meaningfully improved by simple weighting or mixed teachers
- `residual_rzf` and `unfolded_rzf` still remain materially below `WMMSE`
- Sionna is not part of the current mainline acceptance path

## CI

GitHub Actions CI is now defined in [.github/workflows/ci.yml](/home/developer716/workspace/ai-massive-mimo-hybrid-beamforming/.github/workflows/ci.yml). It validates:

- `python -m compileall src scripts tests`
- `pytest -q`

CI does not download DeepMIMO and does not reproduce long training runs or full benchmark artifacts.

After publishing a GitHub release, this repository can be archived through Zenodo/GitHub integration if a DOI is needed.

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
