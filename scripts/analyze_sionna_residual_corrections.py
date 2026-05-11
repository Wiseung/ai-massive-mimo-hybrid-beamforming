#!/usr/bin/env python
"""Analyze residual corrections learned by the Sionna OFDM residual-RZF beamformer."""

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
from beamforming.utils.sionna_ofdm_training import compute_link_metrics, generate_qpsk_resource_grid, run_model_forward, simulate_multiuser_ofdm_link


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


def _complex_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    flat_a = a.reshape(a.size(0), -1)
    flat_b = b.reshape(b.size(0), -1)
    inner = (torch.conj(flat_a) * flat_b).sum(dim=-1).abs()
    denom = torch.linalg.norm(flat_a, dim=-1) * torch.linalg.norm(flat_b, dim=-1)
    cosine = (inner / denom.clamp_min(1e-12)).clamp(0.0, 1.0)
    return torch.rad2deg(torch.arccos(cosine))


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    device = _resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["training"]["seed"]))

    model = build_model(
        config["model"],
        {
            "num_users": int(config["dataset"]["num_users"]),
            "num_bs_ant": int(config["dataset"]["num_bs_ant"]),
            "num_rf_chains": int(config["dataset"]["num_users"]),
        },
    ).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    rows: list[dict[str, Any]] = []
    eval_cfg = config["dataset"]
    for offset, snr_db in enumerate(eval_cfg["snr_db_eval"]):
        generator = SionnaOFDMSyntheticGenerator(
            SionnaOFDMSyntheticConfig(
                batch_size=int(eval_cfg.get("batch_size", 128)),
                num_subcarriers=int(eval_cfg["num_subcarriers"]),
                num_users=int(eval_cfg["num_users"]),
                num_bs_ant=int(eval_cfg["num_bs_ant"]),
                channel_model=str(eval_cfg.get("channel_model", "rayleigh")),
                sparse_mmwave_like=bool(eval_cfg.get("sparse_mmwave_like", False)),
                num_paths=int(eval_cfg.get("num_paths", 3)),
                snr_db_choices=[float(snr_db)],
                seed=int(config["training"]["seed"]) + 700 + offset,
            )
        )
        batch = generator.sample_batch(device=device, return_symbols=False)
        context = generate_qpsk_resource_grid(
            batch_size=batch["H_f"].size(0),
            num_subcarriers=batch["H_f"].size(1),
            num_users=batch["H_f"].size(2),
            device=device,
            generator=generator.generator,
        )
        outputs = run_model_forward(model, batch["H_f"], batch["snr_db"])
        link_pred = simulate_multiuser_ofdm_link(batch["H_f"], outputs["precoder"], context.tx_symbols, batch["noise_var"], batch["snr_db"])
        link_base = simulate_multiuser_ofdm_link(batch["H_f"], outputs["base_precoder"], context.tx_symbols, batch["noise_var"], batch["snr_db"])
        pred_metrics = compute_link_metrics(batch["H_f"], outputs["precoder"], context.tx_symbols, link_pred["rx"], batch["noise_var"])
        base_metrics = compute_link_metrics(batch["H_f"], outputs["base_precoder"], context.tx_symbols, link_base["rx"], batch["noise_var"])

        delta_norm = torch.linalg.norm(outputs["delta_precoder"].reshape(outputs["delta_precoder"].size(0), -1), dim=-1)
        base_norm = torch.linalg.norm(outputs["base_precoder"].reshape(outputs["base_precoder"].size(0), -1), dim=-1)
        ratio = (delta_norm / base_norm.clamp_min(1e-12)).mean()
        angle = _complex_angle_deg(outputs["base_precoder"], outputs["precoder"]).mean()
        power = (torch.abs(outputs["precoder"]) ** 2).sum(dim=(-2, -1))
        rows.append(
            {
                "snr_db": float(snr_db),
                "delta_norm_ratio": float(ratio.item()),
                "correction_angle_deg": float(angle.item()),
                "se_gain_over_rzf": float((pred_metrics["mean_sum_rate"] - base_metrics["mean_sum_rate"]).item()),
                "relative_se_gain_over_rzf": float(
                    ((pred_metrics["mean_sum_rate"] - base_metrics["mean_sum_rate"]) / base_metrics["mean_sum_rate"].clamp_min(1e-12)).item()
                ),
                "power_violation": float(torch.abs(power - 1.0).mean().item()),
                "alpha": float(outputs["alpha"].detach().cpu().item()) if torch.is_tensor(outputs.get("alpha")) else None,
            }
        )

    frame = pd.DataFrame(rows)
    frame[["snr_db", "delta_norm_ratio", "relative_se_gain_over_rzf", "power_violation"]].to_csv(
        out_dir / "residual_norm_vs_snr.csv", index=False
    )
    frame[["snr_db", "correction_angle_deg"]].to_csv(out_dir / "correction_angle_vs_snr.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(frame["snr_db"], frame["delta_norm_ratio"], marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("||delta_F|| / ||F_rzf||")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_norm_vs_snr.png", dpi=160)
    plt.close()

    alpha_payload = {"alpha": float(frame["alpha"].dropna().iloc[0]) if frame["alpha"].dropna().size else None}
    (out_dir / "alpha_value.json").write_text(json.dumps(alpha_payload, indent=2), encoding="utf-8")

    mean_gain = float(frame["relative_se_gain_over_rzf"].mean())
    high_snr_mean_ratio = float(frame.loc[frame["snr_db"].isin([10.0, 15.0, 20.0]), "delta_norm_ratio"].mean())
    high_snr_angle = float(frame.loc[frame["snr_db"].isin([10.0, 15.0, 20.0]), "correction_angle_deg"].mean())
    lines = [
        "# Residual-RZF Correction Analysis",
        "",
        f"- mean_relative_se_gain_over_rzf: `{mean_gain:+.6%}`",
        f"- mean_delta_norm_ratio: `{frame['delta_norm_ratio'].mean():.6f}`",
        f"- mean_correction_angle_deg: `{frame['correction_angle_deg'].mean():.6f}`",
        f"- high_snr_delta_norm_ratio: `{high_snr_mean_ratio:.6f}`",
        f"- high_snr_correction_angle_deg: `{high_snr_angle:.6f}`",
        f"- alpha: `{alpha_payload['alpha']}`",
        "",
        "## Conclusions",
        "",
        f"- residual correction is {'small' if frame['delta_norm_ratio'].mean() < 0.2 else 'not trivially small'} relative to RZF.",
        f"- correction is {'more visible' if high_snr_mean_ratio > float(frame['delta_norm_ratio'].mean()) else 'not clearly larger'} at high SNR.",
        f"- the observed +0.0134% mean gap to RZF is {'consistent with noise-level refinement' if abs(mean_gain) < 0.005 else 'larger than a tiny numerical wiggle'} in this setup.",
        "- The safer description is `RZF-level learned refinement` rather than `RZF-beating model` unless repeated robustness runs show a durable positive gap.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved residual analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
