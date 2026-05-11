#!/usr/bin/env python
"""Compare Sionna channel-extraction artifacts."""

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
    parser.add_argument("--extract", required=True)
    parser.add_argument("--beamforming", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_summary = json.loads(Path(args.extract).read_text(encoding="utf-8"))
    beamforming_summary = json.loads(Path(args.beamforming).read_text(encoding="utf-8"))
    metrics = pd.read_csv(args.metrics)

    comparison_rows = [
        {
            "artifact": "extract_h_f_demo",
            "extraction_success": extract_summary["extraction_success"],
            "project_h_f_shape_compatible": extract_summary["project_h_f_shape_compatible"],
            "used_for_project_rzf": extract_summary["used_for_project_rzf"],
        },
        {
            "artifact": "native_channel_beamforming",
            "extraction_success": beamforming_summary["extraction_success"],
            "project_h_f_assisted": beamforming_summary["project_h_f_assisted"],
            "native_receiver_success": beamforming_summary["native_receiver_success"],
        },
    ]
    with (out_dir / "channel_extraction_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in comparison_rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(comparison_rows)

    ok = metrics[metrics["native_receiver_success"] == True].copy()  # noqa: E712
    if not ok.empty:
        plt.figure(figsize=(8, 4.5))
        plt.bar(ok["method"], ok["approximate_sum_rate"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("approximate_sum_rate")
        plt.tight_layout()
        plt.savefig(out_dir / "sum_rate_by_method.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.bar(ok["method"], ok["symbol_mse"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("symbol_mse")
        plt.tight_layout()
        plt.savefig(out_dir / "mse_by_method.png", dpi=160)
        plt.close()

    lines = [
        "# Sionna Channel Extraction Comparison",
        "",
        f"1. success extracting H_f from Sionna channel tensor: `{extract_summary['extraction_success']}`",
        f"2. extracted H_f compatible with project precoder interface: `{extract_summary['project_h_f_shape_compatible']}`",
        f"3. native-channel-assisted chain success: `{beamforming_summary['native_receiver_success']}`",
        f"4. project_h_f_assisted limitation reduced: `{not beamforming_summary['project_h_f_assisted']}`",
        f"5. learned_residual_rzf can use extracted H_f: `{bool((metrics['method'] == 'learned_residual_rzf_from_extracted_h').any() and metrics[metrics['method'] == 'learned_residual_rzf_from_extracted_h']['extracted_h_f_used'].iloc[0])}`",
        "6. full native-only benchmark achieved: `False`",
        "7. next step: keep reducing project-assisted assumptions on the channel/precoder side while preserving the native Sionna receiver path.",
        "",
        "## Notes",
        "- Current interpretation remains synthetic and optional-Sionna only.",
        "- Even with extracted H_f, this should not yet be described as a full native-only benchmark unless the full channel/precoder/receiver stack is consistently native.",
    ]
    (out_dir / "channel_extraction_comparison.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved channel extraction comparison to {out_dir}")


if __name__ == "__main__":
    main()
