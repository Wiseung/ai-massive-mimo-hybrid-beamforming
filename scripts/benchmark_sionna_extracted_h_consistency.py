#!/usr/bin/env python
"""Benchmark consistency between extracted-H proxy metrics and native receiver metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import (
    compute_project_precoder_per_subcarrier,
    evaluate_ofdm_beamforming_outputs,
    time_function,
)
from beamforming.utils.sionna_native_chain import write_markdown
from beamforming.utils.sionna_native_learned_beamforming import (
    build_native_receiver_context,
    default_checkpoint_path,
    generate_shared_sionna_channel_bundle,
    infer_learned_precoder,
    load_learned_beamformer_checkpoint,
    run_native_receiver_with_precoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--snrs", nargs="+", type=float, default=[0.0, 5.0, 10.0, 15.0, 20.0])
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _native_method_name(method: str) -> str:
    return method.removesuffix("_from_extracted_h")


def _method_type(method: str) -> str:
    return "learned" if method.startswith("learned_") else "analytic"


def _make_md(summary_lines: list[str]) -> str:
    return "\n".join(summary_lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_sionna_env_info()
    summary_path = out_dir / "summary.md"
    metrics_path = out_dir / "metrics.csv"
    rank_path = out_dir / "method_rank_by_snr_seed.csv"
    agreement_path = out_dir / "rank_agreement_table.csv"

    if not env["sionna_import_ok"]:
        write_markdown(
            summary_path,
            [
                "# Extracted H_f Consistency Benchmark",
                "",
                "- status: `skipped`",
                "- reason: `Sionna not installed`",
            ],
        )
        print(f"Saved extracted-H consistency summary to {summary_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[1]
    batch_size = 4 if args.quick else 16
    methods = [
        "project_rzf_from_extracted_h",
        "project_wmmse_iter_2_from_extracted_h",
        "project_wmmse_iter_5_from_extracted_h",
        "learned_residual_rzf_from_extracted_h",
        "learned_residual_wmmse_distill_from_extracted_h",
    ]

    learned_cache: dict[str, Any] = {}
    for method in methods:
        if method.startswith("learned_"):
            ckpt = default_checkpoint_path(_native_method_name(method), repo_root)
            if ckpt.exists():
                learned_cache[method] = load_learned_beamformer_checkpoint(ckpt, device, method_name=_native_method_name(method))

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for snr_db in args.snrs:
            _set_seed(seed)
            noise_var = float(10.0 ** (-float(snr_db) / 10.0))
            channel_bundle = generate_shared_sionna_channel_bundle(
                batch_size=batch_size,
                num_subcarriers=16,
                num_users=4,
                num_bs_ant=16,
                noise_var=noise_var,
                device=device,
                selected_ofdm_symbol="first_data",
                effective_subcarriers="all_effective",
                normalize_channel=False,
            )
            context = build_native_receiver_context(
                batch_size=batch_size,
                num_subcarriers=16,
                num_users=4,
                num_bs_ant=16,
                snr_db=float(snr_db),
                device=device,
                channel_bundle=channel_bundle,
            )
            for method in methods:
                method_type = _method_type(method)
                native_name = _native_method_name(method)
                checkpoint_path = None
                teacher_flag = False
                if method.startswith("project_"):
                    precoder_f, runtime_ms = time_function(
                        compute_project_precoder_per_subcarrier,
                        native_name.removeprefix("project_"),
                        context.h_f,
                        context.noise_var,
                    )
                else:
                    bundle = learned_cache.get(method)
                    if bundle is None:
                        rows.append(
                            {
                                "seed": seed,
                                "snr_db": float(snr_db),
                                "method": method,
                                "method_type": method_type,
                                "extraction_success": False,
                                "native_receiver_success": False,
                                "proxy_sum_rate": float("nan"),
                                "native_sum_rate_or_proxy_receiver_sum_rate": float("nan"),
                                "symbol_mse": float("nan"),
                                "ber_if_available": None,
                                "method_rank_proxy": None,
                                "method_rank_native": None,
                                "rank_agreement": None,
                                "fallback_used": True,
                                "fallback_reason": "skipped_missing_checkpoint",
                            }
                        )
                        continue
                    snr_tensor = torch.full((context.h_f.size(0),), float(snr_db), dtype=torch.float32, device=device)
                    precoder_f, infer_meta, runtime_ms = infer_learned_precoder(
                        bundle,
                        context.h_f,
                        snr_tensor,
                        native_receiver_path=True,
                    )
                    teacher_flag = bool(infer_meta["teacher_used_during_inference"])
                    checkpoint_path = str(bundle.checkpoint_path)

                proxy = evaluate_ofdm_beamforming_outputs(context.h_f, precoder_f, context.stream_symbols, context.noise_var)
                native_row, _, _ = run_native_receiver_with_precoder(
                    method=method,
                    method_type=method_type,
                    precoder_f=precoder_f,
                    context=context,
                    runtime_ms=runtime_ms,
                    checkpoint_path=checkpoint_path,
                    teacher_used_during_inference=teacher_flag,
                    trace_shapes=False,
                )
                native_metric = (
                    float(native_row["approximate_sum_rate"])
                    if native_row["native_receiver_success"]
                    else float(proxy["approximate_sum_rate"])
                )
                rows.append(
                    {
                        "seed": seed,
                        "snr_db": float(snr_db),
                        "method": method,
                        "method_type": method_type,
                        "extraction_success": bool(not context.context_meta.get("project_h_f_assisted", True)),
                        "native_receiver_success": bool(native_row["native_receiver_success"]),
                        "proxy_sum_rate": float(proxy["approximate_sum_rate"]),
                        "native_sum_rate_or_proxy_receiver_sum_rate": native_metric,
                        "symbol_mse": float(native_row["symbol_mse"]) if native_row["native_receiver_success"] else float(proxy["symbol_mse"]),
                        "ber_if_available": native_row["ber_if_available"],
                        "method_rank_proxy": None,
                        "method_rank_native": None,
                        "rank_agreement": None,
                        "fallback_used": bool(native_row["fallback_used"]),
                        "fallback_reason": str(native_row["fallback_reason"]),
                    }
                )

    frame = pd.DataFrame(rows)
    frame["method_rank_proxy"] = (
        frame.groupby(["seed", "snr_db"])["proxy_sum_rate"].rank(method="dense", ascending=False)
    )
    frame["method_rank_native"] = (
        frame.groupby(["seed", "snr_db"])["native_sum_rate_or_proxy_receiver_sum_rate"].rank(method="dense", ascending=False)
    )
    valid_rank_mask = frame["method_rank_proxy"].notna() & frame["method_rank_native"].notna()
    frame.loc[valid_rank_mask, "rank_agreement"] = (
        frame.loc[valid_rank_mask, "method_rank_proxy"] == frame.loc[valid_rank_mask, "method_rank_native"]
    )
    frame.to_csv(metrics_path, index=False)

    rank_frame = frame[
        [
            "seed",
            "snr_db",
            "method",
            "proxy_sum_rate",
            "native_sum_rate_or_proxy_receiver_sum_rate",
            "method_rank_proxy",
            "method_rank_native",
            "rank_agreement",
        ]
    ].copy()
    rank_frame.to_csv(rank_path, index=False)

    agreement_table = (
        frame.groupby("method")
        .agg(
            rows=("method", "size"),
            extraction_success_rate=("extraction_success", "mean"),
            native_receiver_success_rate=("native_receiver_success", "mean"),
            rank_agreement_rate=("rank_agreement", lambda s: float(pd.Series(s).dropna().mean()) if pd.Series(s).dropna().size else float("nan")),
            proxy_sum_rate_mean=("proxy_sum_rate", "mean"),
            native_sum_rate_mean=("native_sum_rate_or_proxy_receiver_sum_rate", "mean"),
        )
        .reset_index()
    )
    agreement_table.to_csv(agreement_path, index=False)

    ok = frame[frame["native_receiver_success"] == True].copy()  # noqa: E712
    if not ok.empty:
        plt.figure(figsize=(7, 5))
        for method, group in ok.groupby("method"):
            plt.scatter(group["proxy_sum_rate"], group["native_sum_rate_or_proxy_receiver_sum_rate"], label=method, alpha=0.8)
        plt.xlabel("proxy_sum_rate")
        plt.ylabel("native_sum_rate_or_proxy_receiver_sum_rate")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "proxy_vs_native_sum_rate.png", dpi=160)
        plt.close()

    summary_lines = [
        "# Extracted H_f Consistency Benchmark",
        "",
        f"- quick: `{args.quick}`",
        f"- batch_size: `{batch_size}`",
        f"- seeds: `{args.seeds}`",
        f"- snrs_db: `{args.snrs}`",
        f"- extraction_success_all_available_rows: `{bool(frame['extraction_success'].dropna().all())}`",
        f"- native_receiver_success_all_available_rows: `{bool(frame[frame['fallback_reason'] != 'skipped_missing_checkpoint']['native_receiver_success'].all())}`",
        "",
        "## Key answers",
    ]

    overall_rank_agreement = frame["rank_agreement"].dropna()
    summary_lines.append(
        f"1. extracted H_f stable usable: `{bool(frame['extraction_success'].dropna().all())}`"
    )
    summary_lines.append(
        "2. proxy/native method ranking exact-agreement rate: "
        f"`{float(overall_rank_agreement.mean()) if not overall_rank_agreement.empty else float('nan'):.6f}`"
    )

    def _gap_vs(method_a: str, method_b: str) -> tuple[float | None, float | None]:
        merged = frame.pivot_table(
            index=["seed", "snr_db"],
            columns="method",
            values="native_sum_rate_or_proxy_receiver_sum_rate",
            aggfunc="first",
        )
        if method_a not in merged.columns or method_b not in merged.columns:
            return None, None
        gap = (merged[method_a] - merged[method_b]) / merged[method_b].replace(0.0, pd.NA)
        return float(gap.mean()), float((gap > 0).mean())

    gap_rzf, frac_rzf = _gap_vs("learned_residual_rzf_from_extracted_h", "project_rzf_from_extracted_h")
    gap_w5, frac_w5 = _gap_vs("learned_residual_rzf_from_extracted_h", "project_wmmse_iter_5_from_extracted_h")
    if gap_rzf is not None:
        summary_lines.append(
            f"3. learned_residual_rzf vs project_rzf mean gap: `{gap_rzf:.6%}`; positive-fraction across seed/SNR: `{frac_rzf:.6f}`"
        )
    else:
        summary_lines.append("3. learned_residual_rzf vs project_rzf mean gap: `unavailable`")
    if gap_w5 is not None:
        summary_lines.append(
            f"4. learned_residual_rzf vs project_wmmse_iter_5 mean gap: `{gap_w5:.6%}`; positive-fraction across seed/SNR: `{frac_w5:.6f}`"
        )
    else:
        summary_lines.append("4. learned_residual_rzf vs project_wmmse_iter_5 mean gap: `unavailable`")
    summary_lines.append("5. interpretation label should remain: `native-channel-assisted`, not `full native-only benchmark`.")
    write_markdown(summary_path, summary_lines)
    print(f"Saved extracted-H consistency benchmark to {out_dir}")


if __name__ == "__main__":
    main()
