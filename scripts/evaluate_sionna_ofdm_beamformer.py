#!/usr/bin/env python
"""Evaluate optional learned OFDM beamformers against analytic priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

add_src_to_path()

from beamforming.data.sionna_ofdm_synthetic import SionnaOFDMSyntheticConfig, SionnaOFDMSyntheticGenerator
from beamforming.models.factory import build_model
from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_ofdm_training import build_baseline_precoder_stack, compute_link_metrics, generate_qpsk_resource_grid, run_model_forward, simulate_multiuser_ofdm_link


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_eval_generator(cfg: dict[str, Any], snr_db: float, seed: int) -> SionnaOFDMSyntheticGenerator:
    return SionnaOFDMSyntheticGenerator(
        SionnaOFDMSyntheticConfig(
            batch_size=int(cfg.get("batch_size", cfg.get("eval_batch_size", 128))),
            num_subcarriers=int(cfg["num_subcarriers"]),
            num_users=int(cfg["num_users"]),
            num_bs_ant=int(cfg["num_bs_ant"]),
            channel_model=str(cfg.get("channel_model", "rayleigh")),
            sparse_mmwave_like=bool(cfg.get("sparse_mmwave_like", False)),
            num_paths=int(cfg.get("num_paths", 3)),
            snr_db_choices=[float(snr_db)],
            seed=seed,
        )
    )


def _evaluate_method(
    method_name: str,
    learned_model: torch.nn.Module | None,
    dataset_cfg: dict[str, Any],
    snr_db: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    generator = _build_eval_generator(dataset_cfg, float(snr_db), seed)
    total_batches = int(dataset_cfg["num_val_batches"])
    notes: list[str] = []
    used_sionna_ofdm = False
    used_sionna_channel = False
    fallback_used = False
    init_method = None
    num_layers = None
    layer_sum_rate_mean = None
    totals = {
        "mean_sum_rate": 0.0,
        "receive_mse": 0.0,
        "approximate_effective_sinr_db": 0.0,
        "power_violation": 0.0,
    }

    for _ in range(total_batches):
        batch = generator.sample_batch(device=device, return_symbols=False)
        context = generate_qpsk_resource_grid(
            batch_size=batch["H_f"].size(0),
            num_subcarriers=batch["H_f"].size(1),
            num_users=batch["H_f"].size(2),
            device=device,
            generator=generator.generator,
        )
        used_sionna_ofdm = used_sionna_ofdm or context.used_sionna_ofdm
        fallback_used = fallback_used or context.fallback_used
        notes.extend(context.notes)

        if method_name in {"rzf", "wmmse_iter_1", "wmmse_iter_2", "wmmse_iter_5"}:
            precoder = build_baseline_precoder_stack(method_name, batch["H_f"], batch["noise_var"])
        else:
            assert learned_model is not None
            outputs = run_model_forward(learned_model, batch["H_f"], batch["snr_db"])
            precoder = outputs["precoder"]
            init_method = outputs.get("init_method", init_method)
            num_layers = outputs.get("num_layers", num_layers)
            if "layer_sum_rates" in outputs and torch.is_tensor(outputs["layer_sum_rates"]) and outputs["layer_sum_rates"].numel() > 0:
                layer_sum_rate_mean = [float(x) for x in outputs["layer_sum_rates"].mean(dim=0).detach().cpu().tolist()]

        link = simulate_multiuser_ofdm_link(
            channel_f=batch["H_f"],
            precoder=precoder,
            tx_symbols=context.tx_symbols,
            noise_var=batch["noise_var"],
            snr_db=batch["snr_db"],
        )
        used_sionna_channel = used_sionna_channel or bool(link["used_sionna_channel"])
        fallback_used = fallback_used or bool(link["fallback_used"])
        if link["note"]:
            notes.append(str(link["note"]))

        metrics = compute_link_metrics(
            channel_f=batch["H_f"],
            precoder=precoder,
            tx_symbols=context.tx_symbols,
            rx_symbols=link["rx"],
            noise_var=batch["noise_var"],
        )
        totals["mean_sum_rate"] += float(metrics["mean_sum_rate"].item())
        totals["receive_mse"] += float(metrics["receive_mse"].item())
        totals["approximate_effective_sinr_db"] += float(metrics["sinr_db"].item())
        totals["power_violation"] += float(metrics["power_violation"].item())

    denom = max(total_batches, 1)
    return {
        "method": method_name,
        "snr_db": float(snr_db),
        "mean_sum_rate": totals["mean_sum_rate"] / denom,
        "receive_mse": totals["receive_mse"] / denom,
        "approximate_effective_sinr_db": totals["approximate_effective_sinr_db"] / denom,
        "power_violation": totals["power_violation"] / denom,
        "used_sionna_ofdm": used_sionna_ofdm,
        "used_sionna_channel": used_sionna_channel,
        "fallback_used": fallback_used,
        "init_method": init_method,
        "num_layers": num_layers,
        "layer_sum_rate_mean": json.dumps(layer_sum_rate_mean) if layer_sum_rate_mean is not None else "",
        "notes": sorted(set(notes)),
    }


def _save_plot(frame: pd.DataFrame, x: str, y: str, out_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for method, group in frame.groupby("method"):
        plt.plot(group[x], group[y], marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_gap_plot(frame: pd.DataFrame, out_path: Path) -> None:
    learned = frame[
        frame["method"].isin(
            {
                "tiny_neural_beamformer",
                "sionna_ofdm_residual_rzf",
                "sionna_ofdm_unfolded_lite",
                "sionna_ofdm_residual_wmmse_distill",
            }
        )
    ]
    plt.figure(figsize=(7, 4.5))
    for method, group in learned.groupby("method"):
        plt.plot(group["snr_db"], group["gap_to_rzf"], marker="o", label=f"{method} vs RZF")
        plt.plot(group["snr_db"], group["gap_to_wmmse_iter_5"], marker="x", linestyle="--", label=f"{method} vs WMMSE-iter5")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Relative gap")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    env_info = collect_sionna_env_info()
    if not env_info["sionna_import_ok"]:
        raise SystemExit(
            "Sionna is not installed in the current environment. Install the optional dependency with "
            "`pip install sionna-no-rt` before running the OFDM evaluation pipeline."
        )

    device = _resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["training"]["seed"]))

    learned_name = str(config["model"]["name"])
    data_cfg = {
        "num_users": int(config["dataset"]["num_users"]),
        "num_bs_ant": int(config["dataset"]["num_bs_ant"]),
        "num_rf_chains": int(config["dataset"]["num_users"]),
    }
    model = build_model(config["model"], data_cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    methods = [learned_name, "rzf", "wmmse_iter_1", "wmmse_iter_2", "wmmse_iter_5"]
    rows: list[dict[str, Any]] = []
    for offset, snr_db in enumerate(config["dataset"]["snr_db_eval"]):
        for method in methods:
            rows.append(
                _evaluate_method(
                    method_name=method,
                    learned_model=model if method == learned_name else None,
                    dataset_cfg=config["dataset"],
                    snr_db=float(snr_db),
                    device=device,
                    seed=int(config["training"]["seed"]) + 100 * offset + methods.index(method),
                )
            )

    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="snr_db", columns="method", values="mean_sum_rate")
    frame["gap_to_rzf"] = (frame["mean_sum_rate"] - frame["snr_db"].map(pivot["rzf"])) / frame["snr_db"].map(pivot["rzf"])
    frame["gap_to_wmmse_iter_5"] = (frame["mean_sum_rate"] - frame["snr_db"].map(pivot["wmmse_iter_5"])) / frame["snr_db"].map(pivot["wmmse_iter_5"])
    frame.to_csv(out_dir / "metrics.csv", index=False)
    _save_plot(frame, "snr_db", "mean_sum_rate", out_dir / "se_vs_snr.png", "Mean sum-rate (bit/s/Hz)")
    _save_plot(frame, "snr_db", "receive_mse", out_dir / "mse_vs_snr.png", "Receive MSE")
    _save_gap_plot(frame, out_dir / "gap_vs_snr.png")

    learned_only = frame[frame["method"] == learned_name].copy()
    high_snr = learned_only[learned_only["snr_db"].isin([10.0, 15.0, 20.0])]
    summary_lines = [
        "# Sionna OFDM Learned Beamformer Evaluation",
        "",
        f"- Learned method: `{learned_name}`",
        f"- Sionna import OK: `{env_info['sionna_import_ok']}`",
        f"- Sionna version: `{env_info['sionna_version']}`",
        f"- Device: `{device}`",
        f"- Used real Sionna OFDM in evaluation path: `{bool(frame['used_sionna_ofdm'].any())}`",
        f"- Used real Sionna AWGN in evaluation path: `{bool(frame['used_sionna_channel'].any())}`",
        f"- Any fallback used: `{bool(frame['fallback_used'].any())}`",
        f"- teacher_used_during_inference: `False`",
        f"- inference_inputs: `{['H_f', 'F_rzf', 'snr_db'] if learned_name == 'sionna_ofdm_residual_wmmse_distill' else 'model-specific'}`",
        "",
        "## Mean learned results",
        "",
        f"- mean_sum_rate: `{learned_only['mean_sum_rate'].mean():.6f}`",
        f"- receive_mse: `{learned_only['receive_mse'].mean():.6f}`",
        f"- approximate_effective_sinr_db: `{learned_only['approximate_effective_sinr_db'].mean():.6f}`",
        f"- mean_gap_to_rzf: `{learned_only['gap_to_rzf'].mean():+.6%}`",
        f"- mean_gap_to_wmmse_iter_5: `{learned_only['gap_to_wmmse_iter_5'].mean():+.6%}`",
        "",
        "## High-SNR gap",
        "",
        f"- mean_high_snr_gap_to_rzf: `{high_snr['gap_to_rzf'].mean():+.6%}`",
        f"- mean_high_snr_gap_to_wmmse_iter_5: `{high_snr['gap_to_wmmse_iter_5'].mean():+.6%}`",
        "",
        "## Limitations",
        "",
        "- Synthetic OFDM channel only.",
        "- No Sionna RT, no ray tracing, and no 5G NR full stack.",
        "- If the learned method remains below RZF or WMMSE-iter5, that gap is reported directly rather than hidden.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "sionna_import_ok": env_info["sionna_import_ok"],
                "sionna_version": env_info["sionna_version"],
                "device": str(device),
                "model_name": learned_name,
                "used_sionna_ofdm": bool(frame["used_sionna_ofdm"].any()),
                "used_sionna_channel": bool(frame["used_sionna_channel"].any()),
                "fallback_used": bool(frame["fallback_used"].any()),
                "learned_mean_sum_rate": float(learned_only["mean_sum_rate"].mean()),
                "learned_mean_receive_mse": float(learned_only["receive_mse"].mean()),
                "learned_mean_gap_to_rzf": float(learned_only["gap_to_rzf"].mean()),
                "learned_mean_gap_to_wmmse_iter_5": float(learned_only["gap_to_wmmse_iter_5"].mean()),
                "high_snr_gap_to_rzf": float(high_snr["gap_to_rzf"].mean()),
                "high_snr_gap_to_wmmse_iter_5": float(high_snr["gap_to_wmmse_iter_5"].mean()),
                "teacher_used_during_inference": False,
                "inference_inputs": ["H_f", "F_rzf", "snr_db"] if learned_name == "sionna_ofdm_residual_wmmse_distill" else None,
                "init_method": learned_only["init_method"].dropna().iloc[0] if "init_method" in learned_only and learned_only["init_method"].dropna().size else None,
                "num_layers": int(learned_only["num_layers"].dropna().iloc[0]) if "num_layers" in learned_only and learned_only["num_layers"].dropna().size else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
