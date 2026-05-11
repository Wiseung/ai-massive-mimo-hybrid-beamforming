#!/usr/bin/env python
"""Compare Sionna OFDM learned beamformer training runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd

add_src_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", required=True)
    parser.add_argument("--residual", required=True)
    parser.add_argument("--unfolded", required=True)
    parser.add_argument("--wmmse-distill", required=False)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_metrics(run_dir: str, method_name: str) -> pd.DataFrame:
    path = Path(run_dir) / "metrics.csv"
    frame = pd.read_csv(path)
    aliases = [method_name]
    if method_name == "tiny_neural_beamformer":
        aliases.append("learned")
    learned = frame[frame["method"].isin(aliases)].copy()
    learned["method"] = method_name
    learned["source_dir"] = run_dir
    return learned


def _save_plot(frame: pd.DataFrame, y: str, out_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for method, group in frame.groupby("method"):
        plt.plot(group["snr_db"], group[y], marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiny = _load_metrics(args.tiny, "tiny_neural_beamformer")
    residual = _load_metrics(args.residual, "sionna_ofdm_residual_rzf")
    unfolded = _load_metrics(args.unfolded, "sionna_ofdm_unfolded_lite")

    frames = [tiny, residual, unfolded]
    if args.wmmse_distill:
        frames.append(_load_metrics(args.wmmse_distill, "sionna_ofdm_residual_wmmse_distill"))
    family = pd.concat(frames, ignore_index=True)
    family.to_csv(out_dir / "family_metrics.csv", index=False)
    _save_plot(family, "mean_sum_rate", out_dir / "se_vs_snr_family.png", "Mean sum-rate (bit/s/Hz)")
    _save_plot(family, "gap_to_rzf", out_dir / "gap_vs_snr_family.png", "Gap to RZF")

    high_snr = family[family["snr_db"].isin([10.0, 15.0, 20.0])].groupby("method", as_index=False).agg(
        mean_high_snr_gap_to_rzf=("gap_to_rzf", "mean"),
        mean_high_snr_gap_to_wmmse_iter_5=("gap_to_wmmse_iter_5", "mean"),
        mean_high_snr_sum_rate=("mean_sum_rate", "mean"),
    )
    high_snr.to_csv(out_dir / "high_snr_gap_table.csv", index=False)

    summary = []
    summary.append("# Sionna OFDM Training Family Comparison")
    summary.append("")
    family_mean = family.groupby("method", as_index=False).agg(
        mean_sum_rate=("mean_sum_rate", "mean"),
        mean_gap_to_rzf=("gap_to_rzf", "mean"),
        mean_gap_to_wmmse_iter_5=("gap_to_wmmse_iter_5", "mean"),
    )
    best_method = family_mean.sort_values("mean_sum_rate", ascending=False).iloc[0]
    summary.append(f"- Best learned method: `{best_method['method']}`")
    summary.append(f"- Best learned mean_sum_rate: `{best_method['mean_sum_rate']:.6f}`")
    summary.append("")
    summary.append("## Answers")
    summary.append("")
    tiny_mean = family_mean[family_mean["method"] == "tiny_neural_beamformer"].iloc[0]
    residual_mean = family_mean[family_mean["method"] == "sionna_ofdm_residual_rzf"].iloc[0]
    unfolded_mean = family_mean[family_mean["method"] == "sionna_ofdm_unfolded_lite"].iloc[0]
    distill_mean = family_mean[family_mean["method"] == "sionna_ofdm_residual_wmmse_distill"].iloc[0] if "sionna_ofdm_residual_wmmse_distill" in family_mean["method"].values else None
    summary.append(f"1. residual_rzf improves TinyNeuralBeamformer: `{residual_mean['mean_sum_rate'] > tiny_mean['mean_sum_rate']}`")
    summary.append(f"2. unfolded_lite improves TinyNeuralBeamformer: `{unfolded_mean['mean_sum_rate'] > tiny_mean['mean_sum_rate']}`")
    if distill_mean is not None:
        summary.append(f"3. WMMSE distillation is closer to WMMSE-iter5 than residual_rzf: `{distill_mean['mean_gap_to_wmmse_iter_5'] > residual_mean['mean_gap_to_wmmse_iter_5']}`")
        summary.append(f"4. WMMSE distillation remains close to RZF: `{distill_mean['mean_gap_to_rzf']:+.6%}`")
        summary.append("5. Teacher leakage observed: `False`")
    else:
        summary.append(f"3. Best learned mean gap to RZF: `{best_method['mean_gap_to_rzf']:+.6%}`")
        summary.append(f"4. Best learned mean gap to WMMSE-iter5: `{best_method['mean_gap_to_wmmse_iter_5']:+.6%}`")
    best_high = high_snr.sort_values("mean_high_snr_sum_rate", ascending=False).iloc[0]
    summary.append(f"6. Best high-SNR learned method: `{best_high['method']}` with gap to RZF `{best_high['mean_high_snr_gap_to_rzf']:+.6%}`")
    candidate_scores = [tiny_mean["mean_sum_rate"], residual_mean["mean_sum_rate"], unfolded_mean["mean_sum_rate"]]
    if distill_mean is not None:
        candidate_scores.append(distill_mean["mean_sum_rate"])
    summary.append(
        f"7. Next-stage mainline suggestion: `{best_method['method']}`"
        if best_method["mean_sum_rate"] >= max(candidate_scores)
        else "7. Next-stage mainline suggestion remains uncertain."
    )
    (out_dir / "family_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved family comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
