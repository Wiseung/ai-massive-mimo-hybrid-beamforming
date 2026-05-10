#!/usr/bin/env python
"""Generate a manifest for key benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ARTIFACTS = [
    ("outputs/comparisons/latency/latency_table.csv", "python scripts/benchmark_latency.py ..."),
    ("outputs/comparisons/latency_v2/latency_table.csv", "python scripts/benchmark_latency.py ... --profile-method unfolded_wmmse_lite"),
    ("outputs/comparisons/latency_v2/unfolded_wmmse_lite_profile.json", "python scripts/benchmark_latency.py ... --profile-method unfolded_wmmse_lite"),
    ("outputs/comparisons/model_families_v3/model_family_table_v3.csv", "python scripts/compare_model_families.py ..."),
    ("outputs/comparisons/deepmimo_model_family_random/deepmimo_model_family_mean_std.csv", "python scripts/run_deepmimo_model_family_benchmark.py --split-mode random ..."),
    ("outputs/comparisons/deepmimo_model_family_contiguous/deepmimo_model_family_mean_std.csv", "python scripts/run_deepmimo_model_family_benchmark.py --split-mode contiguous ..."),
    ("outputs/comparisons/deepmimo_model_family_random_vs_contiguous.csv", "python scripts/check_deepmimo_results.py ..."),
    ("outputs/comparisons/unfolded_wmmse_lite_sweep/sweep_table.csv", "python scripts/sweep_unfolded_wmmse_lite.py ..."),
    ("outputs/comparisons/unfolded_wmmse_lite_sweep/best_variant.yaml", "python scripts/sweep_unfolded_wmmse_lite.py ..."),
    ("outputs/comparisons/model_families_v4/model_family_table.csv", "python scripts/compare_model_families.py --latency-table outputs/comparisons/latency_v2/latency_table.csv --out outputs/comparisons/model_families_v4"),
    ("outputs/comparisons/model_families_v4/pareto_se_latency.png", "python scripts/compare_model_families.py --latency-table outputs/comparisons/latency_v2/latency_table.csv --out outputs/comparisons/model_families_v4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_json.with_suffix(".md")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    rows = []
    for path_str, command in ARTIFACTS:
        path = Path(path_str)
        rows.append(
            {
                "path": path_str,
                "exists": path.exists(),
                "type": path.suffix.lstrip("."),
                "command": command,
                "generated_from_commit": commit,
            }
        )

    payload = {
        "generated_from_commit": commit,
        "note": "Artifact manifest is an index of generated benchmark outputs. It is not a dataset archive and does not imply that raw DeepMIMO data or training checkpoints are committed to git.",
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: this is a result index, not a dataset archive.",
        "",
        "| path | exists | type | command |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['path']} | {row['exists']} | {row['type']} | `{row['command']}` |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
