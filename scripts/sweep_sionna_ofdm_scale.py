#!/usr/bin/env python
"""Quick scale sweep for optional Sionna OFDM residual-RZF experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd
import torch

add_src_to_path()

from beamforming.data.sionna_ofdm_synthetic import SionnaOFDMSyntheticConfig, SionnaOFDMSyntheticGenerator
from beamforming.models.factory import build_model
from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_ofdm_training import build_baseline_precoder_stack, compute_link_metrics, generate_qpsk_resource_grid, run_model_forward, simulate_multiuser_ofdm_link


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _evaluate_method(
    method: str,
    batch: dict[str, torch.Tensor],
    context: Any,
    device: torch.device,
    model: torch.nn.Module | None = None,
) -> dict[str, float]:
    if method in {"rzf", "wmmse_iter_5"}:
        precoder = build_baseline_precoder_stack(method, batch["H_f"], batch["noise_var"])
    else:
        assert model is not None
        precoder = run_model_forward(model, batch["H_f"], batch["snr_db"])["precoder"]
    link = simulate_multiuser_ofdm_link(batch["H_f"], precoder, context.tx_symbols, batch["noise_var"], batch["snr_db"])
    metrics = compute_link_metrics(batch["H_f"], precoder, context.tx_symbols, link["rx"], batch["noise_var"])
    return {
        "mean_sum_rate": float(metrics["mean_sum_rate"].item()),
        "receive_mse": float(metrics["receive_mse"].item()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    rows: list[dict[str, Any]] = []
    scales = {
        "num_subcarriers": [4, 8, 16],
        "num_users": [2, 4],
        "num_bs_ant": [8, 16, 32],
    }
    base_snr = [10.0]
    for nsc in scales["num_subcarriers"]:
        for k in scales["num_users"]:
            for nt in scales["num_bs_ant"]:
                record_base = {
                    "quick": bool(args.quick),
                    "num_subcarriers": nsc,
                    "num_users": k,
                    "num_bs_ant": nt,
                }
                if nsc >= 16 and nt >= 32 and args.quick:
                    for method in ["rzf", "wmmse_iter_5", "sionna_ofdm_residual_rzf"]:
                        rows.append(record_base | {"method": method, "status": "skipped_due_to_runtime"})
                    continue

                generator = SionnaOFDMSyntheticGenerator(
                    SionnaOFDMSyntheticConfig(
                        batch_size=16 if args.quick else 64,
                        num_subcarriers=nsc,
                        num_users=k,
                        num_bs_ant=nt,
                        snr_db_choices=base_snr,
                        seed=1000 + nsc * 100 + k * 10 + nt,
                    )
                )
                batch = generator.sample_batch(device=device, return_symbols=False)
                context = generate_qpsk_resource_grid(
                    batch_size=batch["H_f"].size(0),
                    num_subcarriers=nsc,
                    num_users=k,
                    device=device,
                    generator=generator.generator,
                )
                residual = build_model(
                    {
                        "name": "sionna_ofdm_residual_rzf",
                        "hidden_dim": 128,
                        "alpha_init": 0.1,
                        "learnable_alpha": True,
                        "condition_on_snr": True,
                    },
                    {"num_users": k, "num_bs_ant": nt, "num_rf_chains": k},
                ).to(device)
                residual.eval()

                method_results = {
                    "rzf": _evaluate_method("rzf", batch, context, device),
                    "wmmse_iter_5": _evaluate_method("wmmse_iter_5", batch, context, device),
                    "sionna_ofdm_residual_rzf": _evaluate_method("sionna_ofdm_residual_rzf", batch, context, device, residual),
                }
                rzf_rate = method_results["rzf"]["mean_sum_rate"]
                wmmse_rate = method_results["wmmse_iter_5"]["mean_sum_rate"]
                for method, metrics in method_results.items():
                    rows.append(
                        record_base
                        | {
                            "method": method,
                            "status": "ok",
                            "mean_sum_rate": metrics["mean_sum_rate"],
                            "receive_mse": metrics["receive_mse"],
                            "gap_to_rzf": (metrics["mean_sum_rate"] - rzf_rate) / max(rzf_rate, 1e-12),
                            "gap_to_wmmse_iter_5": (metrics["mean_sum_rate"] - wmmse_rate) / max(wmmse_rate, 1e-12),
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "scale_sweep.csv", index=False)

    ok = frame[frame["status"] == "ok"].copy()
    ok["scale_key"] = ok.apply(lambda row: f"Nsc={int(row['num_subcarriers'])},K={int(row['num_users'])},Nt={int(row['num_bs_ant'])}", axis=1)
    plt.figure(figsize=(10, 5))
    for method, group in ok.groupby("method"):
        plt.plot(group["scale_key"], group["mean_sum_rate"], marker="o", label=method)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean sum-rate (bit/s/Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "se_vs_scale.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    learned = ok[ok["method"] == "sionna_ofdm_residual_rzf"]
    plt.plot(learned["scale_key"], learned["gap_to_rzf"], marker="o", label="residual_rzf vs RZF")
    plt.plot(learned["scale_key"], learned["gap_to_wmmse_iter_5"], marker="x", linestyle="--", label="residual_rzf vs WMMSE-iter5")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Relative gap")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gap_vs_scale.png", dpi=160)
    plt.close()

    lines = [
        "# Sionna OFDM Scale Sweep",
        "",
        f"- quick_mode: `{bool(args.quick)}`",
        f"- skipped_configs: `{int((frame['status'] == 'skipped_due_to_runtime').sum())}`",
        f"- residual mean gap to RZF across evaluated configs: `{learned['gap_to_rzf'].mean():+.6%}`",
        f"- residual mean gap to WMMSE-iter5 across evaluated configs: `{learned['gap_to_wmmse_iter_5'].mean():+.6%}`",
        "",
        "## Notes",
        "",
        "- This is a synthetic-OFDM quick sweep, not a full scale benchmark.",
        "- Skipped configurations are marked explicitly instead of forcing unstable runtime behavior.",
        "- Residual-RZF remains a refinement around the RZF operating point unless a consistent positive gap is observed.",
    ]
    (out_dir / "scale_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved scale sweep outputs to {out_dir}")


if __name__ == "__main__":
    main()
