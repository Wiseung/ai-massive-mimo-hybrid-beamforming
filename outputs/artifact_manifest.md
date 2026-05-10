# Artifact Manifest

- commit: `904d94dffa7c54d7499e8457398e1909b2aeff09`

| path | exists | type | command |
| --- | --- | --- | --- |
| outputs/comparisons/latency/latency_table.csv | True | csv | `python scripts/benchmark_latency.py ...` |
| outputs/comparisons/latency_v2/latency_table.csv | True | csv | `python scripts/benchmark_latency.py ... --profile-method unfolded_wmmse_lite` |
| outputs/comparisons/latency_v2/unfolded_wmmse_lite_profile.json | True | json | `python scripts/benchmark_latency.py ... --profile-method unfolded_wmmse_lite` |
| outputs/comparisons/model_families_v3/model_family_table_v3.csv | True | csv | `python scripts/compare_model_families.py ...` |
| outputs/comparisons/deepmimo_model_family_random/deepmimo_model_family_mean_std.csv | True | csv | `python scripts/run_deepmimo_model_family_benchmark.py --split-mode random ...` |
| outputs/comparisons/deepmimo_model_family_contiguous/deepmimo_model_family_mean_std.csv | True | csv | `python scripts/run_deepmimo_model_family_benchmark.py --split-mode contiguous ...` |
| outputs/comparisons/deepmimo_model_family_random_vs_contiguous.csv | True | csv | `python scripts/check_deepmimo_results.py ...` |
| outputs/comparisons/unfolded_wmmse_lite_sweep/sweep_table.csv | True | csv | `python scripts/sweep_unfolded_wmmse_lite.py ...` |
| outputs/comparisons/unfolded_wmmse_lite_sweep/best_variant.yaml | True | yaml | `python scripts/sweep_unfolded_wmmse_lite.py ...` |
| outputs/comparisons/model_families_v4/model_family_table.csv | True | csv | `python scripts/compare_model_families.py --latency-table outputs/comparisons/latency_v2/latency_table.csv --out outputs/comparisons/model_families_v4` |
| outputs/comparisons/model_families_v4/pareto_se_latency.png | True | png | `python scripts/compare_model_families.py --latency-table outputs/comparisons/latency_v2/latency_table.csv --out outputs/comparisons/model_families_v4` |
