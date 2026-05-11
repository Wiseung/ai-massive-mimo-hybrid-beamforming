#!/usr/bin/env python
"""Run a lightweight SNR mini benchmark for native-chain learned beamformers."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import compute_project_precoder_per_subcarrier, time_function
from beamforming.utils.sionna_native_learned_beamforming import (
    build_native_receiver_context,
    default_checkpoint_path,
    infer_learned_precoder,
    load_learned_beamformer_checkpoint,
    run_native_receiver_with_precoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_sionna_env_info()
    if not env["sionna_import_ok"]:
        raise SystemExit("Sionna optional dependency is not installed.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[1]

    rows = []
    methods = ["project_rzf", "project_wmmse_iter_5", "learned_residual_rzf", "learned_residual_wmmse_distill"]
    for snr_db in [0.0, 5.0, 10.0, 15.0, 20.0]:
        context = build_native_receiver_context(
            batch_size=16,
            num_subcarriers=16,
            num_users=4,
            num_bs_ant=16,
            snr_db=snr_db,
            device=device,
        )
        for method in methods:
            if method.startswith("project_"):
                precoder_f, runtime_ms = time_function(compute_project_precoder_per_subcarrier, method.removeprefix("project_"), context.h_f, context.noise_var)
                teacher_flag = False
                ckpt_path = None
            else:
                ckpt_path = default_checkpoint_path(method, repo_root)
                if not ckpt_path.exists():
                    rows.append(
                        {
                            "snr_db": snr_db,
                            "method": method,
                            "native_receiver_success": False,
                            "teacher_used_during_inference": False,
                            "fallback_used": True,
                            "fallback_stage": "checkpoint",
                            "fallback_reason": "skipped_missing_checkpoint",
                            "symbol_mse": float("nan"),
                            "effective_sinr_db": float("nan"),
                            "approximate_sum_rate": float("nan"),
                        }
                    )
                    continue
                bundle = load_learned_beamformer_checkpoint(ckpt_path, device, method_name=method)
                snr_tensor = torch.full((context.h_f.size(0),), context.snr_db, dtype=torch.float32, device=device)
                precoder_f, infer_meta, runtime_ms = infer_learned_precoder(bundle, context.h_f, snr_tensor, native_receiver_path=True)
                teacher_flag = bool(infer_meta["teacher_used_during_inference"])
            result, _, _ = run_native_receiver_with_precoder(
                method=method,
                method_type="learned" if method.startswith("learned_") else "analytic",
                precoder_f=precoder_f,
                context=context,
                runtime_ms=runtime_ms,
                checkpoint_path=str(ckpt_path) if ckpt_path is not None else None,
                teacher_used_during_inference=teacher_flag,
                trace_shapes=False,
            )
            result["snr_db"] = snr_db
            rows.append(result)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "metrics.csv", index=False)
    plt.figure(figsize=(7, 4.5))
    for method, group in frame.groupby("method"):
        plt.plot(group["snr_db"], group["symbol_mse"], marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel("symbol_mse")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "mse_vs_snr.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for method, group in frame.groupby("method"):
        plt.plot(group["snr_db"], group["effective_sinr_db"], marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel("effective_sinr_db")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "sinr_vs_snr.png", dpi=160)
    plt.close()

    if {"project_rzf", "project_wmmse_iter_5"}.issubset(set(frame["method"])):
        ref = frame[frame["method"] == "project_rzf"][["snr_db", "approximate_sum_rate"]].rename(columns={"approximate_sum_rate": "ref"})
        gap = frame.merge(ref, on="snr_db", how="left")
        gap["gap_to_project_rzf"] = (gap["approximate_sum_rate"] - gap["ref"]) / gap["ref"]
        plt.figure(figsize=(7, 4.5))
        for method, group in gap.groupby("method"):
            plt.plot(group["snr_db"], group["gap_to_project_rzf"], marker="o", label=method)
        plt.xlabel("SNR (dB)")
        plt.ylabel("gap_to_project_rzf")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "gap_vs_snr.png", dpi=160)
        plt.close()
    else:
        gap = frame.copy()

    summary_lines = [
        "# Native Learned Mini Benchmark",
        "",
        "- This is a lightweight checkpoint-only SNR sweep.",
        "- The receiver chain is native Sionna.",
        "- The precoder/H_f side is still project-assisted.",
        "- This is not a full native-only benchmark.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved native learned mini benchmark to {out_dir}")


if __name__ == "__main__":
    main()
