#!/usr/bin/env python
"""Benchmark inference latency and parameter counts for optional Sionna OFDM models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd
import torch

add_src_to_path()

from beamforming.data.sionna_ofdm_synthetic import SionnaOFDMSyntheticConfig, SionnaOFDMSyntheticGenerator
from beamforming.metrics.sum_rate import noise_variance_from_snr
from beamforming.models.factory import build_model
from beamforming.utils.sionna_ofdm_training import build_baseline_precoder_stack, compute_link_metrics, run_model_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--timed-runs", type=int, default=100)
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_params(model: torch.nn.Module | None) -> int:
    if model is None:
        return 0
    return int(sum(param.numel() for param in model.parameters()))


def _benchmark_callable(fn: Any, device: torch.device, warmup_runs: int, timed_runs: int) -> tuple[float, float]:
    for _ in range(warmup_runs):
        _ = fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        _ = fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    series = pd.Series(samples)
    return float(series.mean()), float(series.std(ddof=0))


def _make_model_config(name: str) -> dict[str, Any]:
    if name == "tiny_neural_beamformer":
        return {"name": name, "hidden_dim": 128, "condition_on_snr": True, "normalize_power": True}
    if name == "sionna_ofdm_residual_rzf":
        return {"name": name, "hidden_dim": 128, "alpha_init": 0.1, "learnable_alpha": True, "condition_on_snr": True}
    if name == "sionna_ofdm_unfolded_lite":
        return {
            "name": name,
            "hidden_dim": 128,
            "init_method": "wmmse_iter_2",
            "num_layers": 3,
            "learnable_step_size": True,
            "condition_on_snr": True,
        }
    raise ValueError(f"Unsupported learned model for benchmark: {name}")


def _method_flags(method: str) -> tuple[bool, bool, bool]:
    uses_wmmse_init = method.startswith("wmmse_iter_") or method == "wmmse" or method == "sionna_ofdm_unfolded_lite"
    uses_rzf_init = method == "rzf" or method == "sionna_ofdm_residual_rzf"
    trainable = method in {"tiny_neural_beamformer", "sionna_ofdm_residual_rzf", "sionna_ofdm_unfolded_lite"}
    return uses_wmmse_init, uses_rzf_init, trainable


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = SionnaOFDMSyntheticGenerator(
        SionnaOFDMSyntheticConfig(
            batch_size=128,
            num_subcarriers=8,
            num_users=4,
            num_bs_ant=16,
            snr_db_choices=[10.0],
            seed=123,
        )
    )
    batch = generator.sample_batch(device=device, return_symbols=True)
    methods = [
        "rzf",
        "wmmse_iter_1",
        "wmmse_iter_2",
        "wmmse_iter_5",
        "tiny_neural_beamformer",
        "sionna_ofdm_residual_rzf",
        "sionna_ofdm_unfolded_lite",
    ]
    rows: list[dict[str, Any]] = []
    tradeoff_rows: list[dict[str, Any]] = []

    for method in methods:
        if method in {"rzf", "wmmse_iter_1", "wmmse_iter_2", "wmmse_iter_5"}:
            def fn(method_name: str = method) -> torch.Tensor:
                return build_baseline_precoder_stack(method_name, batch["H_f"], batch["noise_var"])

            mean_ms, std_ms = _benchmark_callable(fn, device, args.warmup_runs, args.timed_runs)
            precoder = fn()
            model = None
            init_method = method if method.startswith("wmmse_iter_") else None
        else:
            model_cfg = _make_model_config(method)
            model = build_model(model_cfg, {"num_users": 4, "num_bs_ant": 16, "num_rf_chains": 4}).to(device)
            model.eval()

            def fn() -> torch.Tensor:
                return run_model_forward(model, batch["H_f"], batch["snr_db"])["precoder"]

            mean_ms, std_ms = _benchmark_callable(fn, device, args.warmup_runs, args.timed_runs)
            outputs = run_model_forward(model, batch["H_f"], batch["snr_db"])
            precoder = outputs["precoder"]
            init_method = outputs.get("init_method")

        metrics = compute_link_metrics(
            channel_f=batch["H_f"],
            precoder=precoder,
            tx_symbols=batch["symbols"],
            rx_symbols=torch.matmul(batch["H_f"], torch.matmul(precoder, batch["symbols"].unsqueeze(-1)).squeeze(-1).unsqueeze(-1)).squeeze(-1),
            noise_var=batch["noise_var"],
        )
        uses_wmmse_init, uses_rzf_init, trainable = _method_flags(method)
        row = {
            "method": method,
            "latency_ms_mean": mean_ms,
            "latency_ms_std": std_ms,
            "num_params": _count_params(model),
            "uses_wmmse_init": uses_wmmse_init,
            "uses_rzf_init": uses_rzf_init,
            "trainable": trainable,
            "mean_sum_rate_proxy": float(metrics["mean_sum_rate"].item()),
            "init_method": init_method,
        }
        rows.append(row)
        tradeoff_rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "latency_table.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    ordered = frame.sort_values("latency_ms_mean")
    plt.bar(ordered["method"], ordered["latency_ms_mean"], yerr=ordered["latency_ms_std"], alpha=0.85)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Inference latency (ms)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for _, row in frame.iterrows():
        plt.scatter(row["latency_ms_mean"], row["mean_sum_rate_proxy"], s=80 if row["trainable"] else 60)
        plt.text(row["latency_ms_mean"], row["mean_sum_rate_proxy"], row["method"], fontsize=8)
    plt.xlabel("Inference latency (ms)")
    plt.ylabel("Mean sum-rate proxy (noiseless)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "se_latency_tradeoff.png", dpi=160)
    plt.close()

    notes = {
        "device": str(device),
        "shape": {"B": 128, "Nsc": 8, "K": 4, "Nt": 16},
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "limitation": "This benchmark measures inference latency only; it does not use training time as latency.",
    }
    (out_dir / "latency_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(f"Saved latency benchmark outputs to {out_dir}")


if __name__ == "__main__":
    main()
