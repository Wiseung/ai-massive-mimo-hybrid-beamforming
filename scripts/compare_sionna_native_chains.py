#!/usr/bin/env python
"""Compare the Sionna-native baseline chain and beamforming chain outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd

add_src_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--beamforming", required=True)
    parser.add_argument("--receiver")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    beamforming = json.loads(Path(args.beamforming).read_text(encoding="utf-8"))
    receiver = json.loads(Path(args.receiver).read_text(encoding="utf-8")) if args.receiver else None
    metrics = pd.read_csv(args.metrics)

    comparison_rows = [
        {
            "artifact": "baseline_chain",
            "used_sionna_resource_grid": True,
            "used_sionna_channel": baseline["used_sionna_native_components"],
            "used_sionna_estimator": baseline.get("used_ls_lmmse", False),
            "used_sionna_equalizer": baseline.get("used_ls_lmmse", False),
            "used_sionna_demapper": "Demapper" in baseline["used_components"],
            "fallback_used": baseline["fallback_used"],
        },
        {
            "artifact": "beamforming_chain",
            "used_sionna_resource_grid": beamforming["used_sionna_resource_grid"],
            "used_sionna_channel": beamforming["used_sionna_channel"],
            "used_sionna_estimator": beamforming["used_sionna_estimator"],
            "used_sionna_equalizer": beamforming["used_sionna_equalizer"],
            "used_sionna_demapper": beamforming["used_sionna_demapper"],
            "fallback_used": beamforming["fallback_used"],
        },
    ]
    if receiver is not None:
        comparison_rows.append(
            {
                "artifact": "beamforming_receiver_chain",
                "used_sionna_resource_grid": receiver["used_sionna_resource_grid"],
                "used_sionna_channel": receiver["used_sionna_channel"],
                "used_sionna_estimator": receiver["used_sionna_estimator"],
                "used_sionna_equalizer": receiver["used_sionna_equalizer"],
                "used_sionna_demapper": receiver["used_sionna_demapper"],
                "fallback_used": receiver["fallback_used"],
            }
        )

    comparison_csv = out_dir / ("native_chain_comparison_v2.csv" if receiver is not None else "native_chain_comparison.csv")
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    ok_metrics = metrics[metrics["method"].isin(["project_rzf", "project_wmmse_iter_5", "no_precoding"])].copy()
    plt.figure(figsize=(7, 4))
    plt.bar(ok_metrics["method"], ok_metrics["symbol_mse"])
    plt.ylabel("symbol_mse")
    plt.tight_layout()
    plt.savefig(out_dir / ("mse_by_method_v2.png" if receiver is not None else "mse_by_method.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(ok_metrics["method"], ok_metrics["effective_sinr_db"])
    plt.ylabel("effective_sinr_db")
    plt.tight_layout()
    plt.savefig(out_dir / ("sinr_by_method_v2.png" if receiver is not None else "sinr_by_method.png"), dpi=160)
    plt.close()

    best_method = ok_metrics.sort_values("approximate_sum_rate", ascending=False).iloc[0]
    status_table = out_dir / "receiver_chain_status_table.csv"
    metrics[["method", "used_sionna_estimator", "used_sionna_equalizer", "used_sionna_demapper", "fallback_used", "fallback_stage", "fallback_reason"]].to_csv(status_table, index=False)

    summary_lines = [
        "# Sionna Native Chain Comparison",
        "",
        f"1. receiver chain succeeds under no_precoding: `{bool(receiver and any(row['method']=='no_precoding' and (not row['fallback_used']) for row in receiver['metrics']))}`",
        f"2. receiver chain succeeds under beamforming methods: `{bool(receiver and any(row['method']!='no_precoding' and (not row['fallback_used']) for row in receiver['metrics']))}`",
        f"3. beamforming chain truly uses Sionna ResourceGrid: `{beamforming['used_sionna_resource_grid']}`",
        f"4. beamforming chain truly uses Sionna OFDMChannel/ApplyOFDMChannel path: `{beamforming['used_sionna_channel']}`",
        f"5. project_rzf improves no_precoding approximate sum-rate: `{float(ok_metrics[ok_metrics['method']=='project_rzf']['approximate_sum_rate'].iloc[0]) > float(ok_metrics[ok_metrics['method']=='no_precoding']['approximate_sum_rate'].iloc[0])}`",
        f"6. project_wmmse_iter_5 improves project_rzf approximate sum-rate: `{float(ok_metrics[ok_metrics['method']=='project_wmmse_iter_5']['approximate_sum_rate'].iloc[0]) > float(ok_metrics[ok_metrics['method']=='project_rzf']['approximate_sum_rate'].iloc[0])}`",
        f"7. current best method by approximate sum-rate: `{best_method['method']}`",
        f"8. current learned-beamformer insertion recommendation: `{best_method['method'] in {'project_rzf', 'project_wmmse_iter_5', 'project_wmmse_iter_2'}}`",
        f"9. next step should connect residual_rzf learned model: `{True}`",
        "",
        "## Fallback Notes",
    ]
    for note in (receiver["notes"] if receiver is not None else beamforming["notes"]):
        summary_lines.append(f"- {note}")
    (out_dir / ("native_chain_comparison_v2.md" if receiver is not None else "native_chain_comparison.md")).write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved native chain comparison artifacts to {out_dir}")


if __name__ == "__main__":
    main()
