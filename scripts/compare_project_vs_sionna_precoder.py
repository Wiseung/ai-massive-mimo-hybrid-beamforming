#!/usr/bin/env python
"""Compare project RZF with the probed Sionna native RZF bridge artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = Path(args.probe)
    metrics_path = Path(args.metrics)
    summary = json.loads(probe_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() and metrics_path.stat().st_size > 0 else pd.DataFrame()

    payload: dict[str, Any] = {
        "sionna_rzf_callable": bool(summary.get("sionna_rzf_callable", False)),
        "converted_to_precoder_output": bool(summary.get("converted_to_precoder_output", False)),
        "native_receiver_success_if_attempted": bool(summary.get("native_receiver_success_if_attempted", False)),
        "shape_compatible": bool(summary.get("shape_compatible", False)),
        "power_norm_project": summary.get("power_norm_project"),
        "power_norm_sionna": summary.get("power_norm_sionna"),
        "max_abs_diff_if_comparable": summary.get("max_abs_diff_if_comparable"),
        "sionna_native_precoder_true_now": bool(summary.get("converted_to_precoder_output", False)),
        "full_native_only": False,
        "recommended_next_step": summary.get("recommended_next_step", "keep_project_side_precoder_output"),
    }

    comparison_df = pd.DataFrame(
        [
            {
                "method": "project_rzf_vs_sionna_rzf_precoder",
                "sionna_rzf_callable": payload["sionna_rzf_callable"],
                "converted_to_precoder_output": payload["converted_to_precoder_output"],
                "native_receiver_success_if_attempted": payload["native_receiver_success_if_attempted"],
                "shape_compatible": payload["shape_compatible"],
                "power_norm_project": payload["power_norm_project"],
                "power_norm_sionna": payload["power_norm_sionna"],
                "max_abs_diff_if_comparable": payload["max_abs_diff_if_comparable"],
                "sionna_native_precoder_true_now": payload["sionna_native_precoder_true_now"],
                "full_native_only": False,
                "recommended_next_step": payload["recommended_next_step"],
            }
        ]
    )
    comparison_df.to_csv(out_dir / "project_vs_sionna_precoder_comparison.csv", index=False)

    lines = [
        "# Project vs Sionna Native Precoder Comparison",
        "",
        f"1. Sionna RZFPrecoder callable: `{payload['sionna_rzf_callable']}`",
        f"2. converted to PrecoderOutput: `{payload['converted_to_precoder_output']}`",
        f"3. enters native receiver path: `{payload['native_receiver_success_if_attempted']}`",
        f"4. shape/power close: `shape_compatible={payload['shape_compatible']}`, `power_norm_project={payload['power_norm_project']}`, `power_norm_sionna={payload['power_norm_sionna']}`",
        f"5. sionna_native_precoder=true allowed: `{payload['sionna_native_precoder_true_now']}`",
        "6. full native-only benchmark: `False`",
        f"7. recommended next step: `{payload['recommended_next_step']}`",
        "",
        "Current interpretation:",
        "- this phase establishes API/shape compatibility mapping, not a project-side precoder replacement claim",
        "- if converted_to_precoder_output=true, the adapter bridge is working for the current minimal probe",
        "- the benchmark boundary still remains not full native-only",
    ]
    write_markdown(out_dir / "project_vs_sionna_precoder_comparison.md", lines)
    print(f"Saved project-vs-Sionna precoder comparison to {out_dir}")


if __name__ == "__main__":
    main()
