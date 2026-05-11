#!/usr/bin/env python
"""Evaluate an optional learned OFDM beamformer against analytic baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from beamforming.utils.sionna_ofdm_training import build_baseline_precoder_stack, compute_link_metrics, generate_qpsk_resource_grid, simulate_multiuser_ofdm_link


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


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _evaluate_method(
    method_name: str,
    model: torch.nn.Module | None,
    generator_cfg: dict,
    snr_db: float,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    generator = SionnaOFDMSyntheticGenerator(
        SionnaOFDMSyntheticConfig(
            batch_size=int(generator_cfg["batch_size"]),
            num_subcarriers=int(generator_cfg["num_subcarriers"]),
            num_users=int(generator_cfg["num_users"]),
            num_bs_ant=int(generator_cfg["num_bs_ant"]),
            channel_model=str(generator_cfg.get("channel_model", "rayleigh")),
            sparse_mmwave_like=bool(generator_cfg.get("sparse_mmwave_like", False)),
            num_paths=int(generator_cfg.get("num_paths", 3)),
            snr_db_choices=[float(snr_db)],
            seed=seed,
        )
    )
    total_batches = int(generator_cfg["num_val_batches"])
    notes: list[str] = []
    used_sionna_ofdm = False
    used_sionna_channel = False
    fallback_used = False
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

        if method_name == "learned":
            assert model is not None
            precoder = model(batch["H_f"], snr_db=batch["snr_db"])
        else:
            precoder = build_baseline_precoder_stack(method_name, batch["H_f"], batch["noise_var"])
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

    data_cfg = {
        "num_users": int(config["dataset"]["num_users"]),
        "num_bs_ant": int(config["dataset"]["num_bs_ant"]),
        "num_rf_chains": int(config["dataset"]["num_users"]),
    }
    model = build_model(config["model"], data_cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    rows: list[dict[str, object]] = []
    for offset, snr_db in enumerate(config["dataset"]["snr_db_eval"]):
        rows.append(_evaluate_method("learned", model, config["dataset"], float(snr_db), device, int(config["training"]["seed"]) + 100 * offset))
        rows.append(_evaluate_method("rzf", None, config["dataset"], float(snr_db), device, int(config["training"]["seed"]) + 100 * offset))
        rows.append(_evaluate_method("wmmse_iter_5", None, config["dataset"], float(snr_db), device, int(config["training"]["seed"]) + 100 * offset))

    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="snr_db", columns="method", values="mean_sum_rate")
    if {"learned", "rzf"}.issubset(pivot.columns):
        frame.loc[frame["method"] == "learned", "gap_to_rzf"] = (
            (pivot["learned"] - pivot["rzf"]) / pivot["rzf"]
        ).reindex(frame.loc[frame["method"] == "learned", "snr_db"]).to_numpy()
    if {"learned", "wmmse_iter_5"}.issubset(pivot.columns):
        frame.loc[frame["method"] == "learned", "gap_to_wmmse_iter_5"] = (
            (pivot["learned"] - pivot["wmmse_iter_5"]) / pivot["wmmse_iter_5"]
        ).reindex(frame.loc[frame["method"] == "learned", "snr_db"]).to_numpy()
    frame.to_csv(out_dir / "metrics.csv", index=False)
    _save_plot(frame, "snr_db", "mean_sum_rate", out_dir / "se_vs_snr.png", "Mean sum-rate (bit/s/Hz)")
    _save_plot(frame, "snr_db", "receive_mse", out_dir / "mse_vs_snr.png", "Receive MSE")

    learned_only = frame[frame["method"] == "learned"].copy()
    summary_lines = [
        "# Sionna OFDM Learned Beamformer Evaluation",
        "",
        f"- Sionna import OK: `{env_info['sionna_import_ok']}`",
        f"- Sionna version: `{env_info['sionna_version']}`",
        f"- Device: `{device}`",
        f"- Used real Sionna OFDM in evaluation path: `{bool(frame['used_sionna_ofdm'].any())}`",
        f"- Used real Sionna AWGN in evaluation path: `{bool(frame['used_sionna_channel'].any())}`",
        f"- Any fallback used: `{bool(frame['fallback_used'].any())}`",
        "",
        "## Mean learned results",
        "",
        f"- mean_sum_rate: `{learned_only['mean_sum_rate'].mean():.6f}`",
        f"- receive_mse: `{learned_only['receive_mse'].mean():.6f}`",
        f"- approximate_effective_sinr_db: `{learned_only['approximate_effective_sinr_db'].mean():.6f}`",
        "",
        "## Gap summary",
        "",
    ]
    if "gap_to_rzf" in learned_only:
        summary_lines.append(f"- mean_gap_to_rzf: `{learned_only['gap_to_rzf'].mean():+.6%}`")
    if "gap_to_wmmse_iter_5" in learned_only:
        summary_lines.append(f"- mean_gap_to_wmmse_iter_5: `{learned_only['gap_to_wmmse_iter_5'].mean():+.6%}`")
    summary_lines.extend(
        [
            "",
            "## Scope notes",
            "",
            "- This is an optional synthetic OFDM link-level evaluation only.",
            "- It does not change the v0.1.0 or v0.2.0 benchmark claims.",
            "- It is not Sionna RT, not ray tracing, and not a 5G NR full stack.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "sionna_import_ok": env_info["sionna_import_ok"],
                "sionna_version": env_info["sionna_version"],
                "device": str(device),
                "used_sionna_ofdm": bool(frame["used_sionna_ofdm"].any()),
                "used_sionna_channel": bool(frame["used_sionna_channel"].any()),
                "fallback_used": bool(frame["fallback_used"].any()),
                "learned_mean_sum_rate": float(learned_only["mean_sum_rate"].mean()),
                "learned_mean_receive_mse": float(learned_only["receive_mse"].mean()),
                "learned_mean_gap_to_rzf": float(learned_only["gap_to_rzf"].mean()) if "gap_to_rzf" in learned_only else None,
                "learned_mean_gap_to_wmmse_iter_5": float(learned_only["gap_to_wmmse_iter_5"].mean()) if "gap_to_wmmse_iter_5" in learned_only else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
