#!/usr/bin/env python
"""Sweep unfolded WMMSE-lite variants and summarize SE-latency tradeoffs."""

from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


DEFAULT_LATENCY_PROTOCOL = {
    "batch_size": 512,
    "warmup_runs": 20,
    "timed_runs": 100,
    "include_data_transfer": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_variant_config(
    base_cfg: dict[str, Any],
    *,
    init_method: str,
    num_layers: int,
    distill_weight: float,
    delta_norm_weight: float,
    quick: bool,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["model"]["init_method"] = init_method
    if init_method.startswith("wmmse_iter_"):
        cfg["model"]["init_wmmse_iters"] = int(init_method.removeprefix("wmmse_iter_"))
    cfg["model"]["num_layers"] = num_layers
    cfg.setdefault("loss", {})
    cfg["loss"]["rate_weight"] = 1.0
    cfg["loss"]["distill_weight"] = distill_weight
    cfg["loss"]["teacher"] = "wmmse"
    cfg["loss"]["teacher_max_iter"] = 5 if quick else 30
    cfg["loss"]["delta_norm_weight"] = delta_norm_weight
    if quick:
        cfg["training"]["epochs"] = min(int(cfg["training"].get("epochs", 8)), 4)
        cfg["training"]["batch_size"] = max(int(cfg["training"].get("batch_size", 192)), 256)
    return cfg


def _variant_tag(init_method: str, num_layers: int, distill_weight: float, delta_norm_weight: float) -> str:
    return (
        f"{init_method}_L{num_layers}_dw{distill_weight}_delta{delta_norm_weight}"
        .replace(".", "p")
        .replace("-", "m")
    )


def _candidate_space(quick: bool) -> list[tuple[str, int, float, float]]:
    if quick:
        return [
            ("rzf", 3, 0.1, 1e-3),
            ("wmmse_iter_1", 3, 0.1, 1e-3),
            ("wmmse_iter_2", 3, 0.1, 1e-3),
            ("wmmse_iter_2", 3, 0.0, 1e-3),
            ("wmmse_iter_5", 3, 0.1, 1e-3),
        ]
    return list(
        itertools.product(
            ["rzf", "wmmse_iter_1", "wmmse_iter_2", "wmmse_iter_5"],
            [1, 2, 3, 5],
            [0.0, 0.1, 0.5],
            [0.0, 1e-4, 1e-3],
        )
    )


def _evaluate_variant(
    cfg_path: Path,
    run_dir: Path,
    eval_dir: Path,
    data_path: str,
    device: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not (eval_dir / "summary.yaml").exists():
        _run(
            [
                "python",
                "scripts/evaluate_all.py",
                "--data",
                data_path,
                "--ckpt",
                str(run_dir / "best.pt"),
                "--config",
                str(cfg_path),
                "--methods",
                "rzf",
                "wmmse",
                "wmmse_iter_5",
                "unfolded_wmmse_lite",
                "--out",
                str(eval_dir),
                "--device",
                device,
            ]
        )
    summary = _load_yaml(eval_dir / "summary.yaml")
    curve = pd.read_csv(eval_dir / "synthetic_all_methods.csv")
    return summary, curve


def _compute_gap_to_method(curve: pd.DataFrame, method: str, reference_method: str) -> float:
    method_df = curve[curve["method"] == method][["snr_db", "se"]].rename(columns={"se": "method_se"})
    ref_df = curve[curve["method"] == reference_method][["snr_db", "se"]].rename(columns={"se": "ref_se"})
    merged = method_df.merge(ref_df, on="snr_db", how="inner")
    if merged.empty:
        raise ValueError(f"Reference method {reference_method} is missing from evaluation curve.")
    gap = (merged["method_se"] - merged["ref_se"]) / merged["ref_se"].abs().clip(lower=1e-12)
    return float(gap.mean())


def _latency_variant(
    cfg_path: Path,
    run_dir: Path,
    latency_dir: Path,
    data_path: str,
    device: str,
) -> pd.DataFrame:
    if not (latency_dir / "latency_table.csv").exists():
        _run(
            [
                "python",
                "scripts/benchmark_latency.py",
                "--data",
                data_path,
                "--methods",
                "unfolded_wmmse_lite",
                "--batch-size",
                str(DEFAULT_LATENCY_PROTOCOL["batch_size"]),
                "--warmup-runs",
                str(DEFAULT_LATENCY_PROTOCOL["warmup_runs"]),
                "--timed-runs",
                str(DEFAULT_LATENCY_PROTOCOL["timed_runs"]),
                "--out",
                str(latency_dir),
                "--device",
                device,
                "--artifact-spec",
                f"unfolded_wmmse_lite=unfolded_wmmse_lite,{cfg_path},{run_dir / 'best.pt'}",
            ]
        )
    return pd.read_csv(latency_dir / "latency_table.csv")


def _maybe_train_variant(
    cfg_path: Path,
    run_dir: Path,
    data_path: str,
    device: str,
) -> None:
    if (run_dir / "best.pt").exists():
        return
    _run(
        [
            "python",
            "scripts/train.py",
            "--config",
            str(cfg_path),
            "--data",
            data_path,
            "--out",
            str(run_dir),
            "--device",
            device,
        ]
    )


def _existing_seed_variant_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    default_cfg = Path("configs/synthetic_unfolded_wmmse_lite_iter2.yaml")
    default_run = Path("outputs/runs/synthetic_unfolded_wmmse_lite_iter2")
    default_eval = Path("outputs/comparisons/synthetic_unfolded_wmmse_lite_iter2")
    default_latency = Path("outputs/comparisons/latency_v2/latency_table.csv")
    if default_cfg.exists() and default_run.exists() and default_eval.exists() and default_latency.exists():
        summary = _load_yaml(default_eval / "summary.yaml")
        curve = pd.read_csv(default_eval / "synthetic_all_methods.csv")
        train_summary = _load_yaml(default_run / "train_summary.yaml")
        latency_df = pd.read_csv(default_latency)
        latency_value = float(latency_df.loc[latency_df["method"] == "unfolded_wmmse_lite", "inference_latency_ms"].iloc[0])
        rows.append(
            {
                "tag": "existing_wmmse_iter_2_L5_dw0p1_delta0p01",
                "config_path": str(default_cfg),
                "run_dir": str(default_run),
                "eval_dir": str(default_eval),
                "latency_dir": "outputs/comparisons/latency_v2",
                "source": "existing",
                "init_method": "wmmse_iter_2",
                "num_layers": 5,
                "distill_weight": 0.1,
                "delta_norm_weight": 0.01,
                "mean_se": summary["mean_se"],
                "gap_to_wmmse": summary["mean_gap_to_best_baseline"],
                "gap_to_wmmse_iter_5": _compute_gap_to_method(curve, "unfolded_wmmse_lite", "wmmse_iter_5"),
                "latency_ms": latency_value,
                "num_params": train_summary.get("num_params"),
                "train_time": train_summary.get("train_time_sec"),
            }
        )
    return rows


def _write_outputs(out_dir: Path, table: pd.DataFrame) -> None:
    table = table.sort_values(["mean_se", "latency_ms"], ascending=[False, True]).reset_index(drop=True)
    table.to_csv(out_dir / "sweep_table.csv", index=False)

    md_lines = [
        "# Unfolded WMMSE-lite Sweep",
        "",
        "| tag | source | init | layers | distill | delta | mean_se | gap_to_wmmse | gap_to_wmmse_iter_5 | latency_ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples():
        md_lines.append(
            f"| {row.tag} | {row.source} | {row.init_method} | {row.num_layers} | {row.distill_weight} | {row.delta_norm_weight} | "
            f"{row.mean_se:.6f} | {row.gap_to_wmmse:+.4%} | {row.gap_to_wmmse_iter_5:+.4%} | {row.latency_ms:.3f} |"
        )
    (out_dir / "sweep_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    best_by_se = table.sort_values(["mean_se", "latency_ms"], ascending=[False, True]).iloc[0].to_dict()
    best_pareto_like = table.sort_values(["gap_to_wmmse_iter_5", "latency_ms"], ascending=[False, True]).iloc[0].to_dict()
    best_payload = {
        "selection_note": "best_by_se uses highest mean_se with latency tie-break; best_pareto_like uses closest gap_to_wmmse_iter_5 then lower latency.",
        "latency_protocol": DEFAULT_LATENCY_PROTOCOL,
        "best_by_se": best_by_se,
        "best_pareto_like": best_pareto_like,
    }
    _dump_yaml(out_dir / "best_variant.yaml", best_payload)

    plt.figure(figsize=(8.0, 4.8))
    plt.scatter(table["latency_ms"], table["mean_se"], s=70)
    for row in table.itertuples():
        plt.annotate(row.tag, (row.latency_ms, row.mean_se), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Mean SE")
    plt.title("Unfolded WMMSE-lite Sweep Pareto")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto.png")
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    ordered = table.sort_values("latency_ms")
    plt.plot(ordered["latency_ms"], ordered["mean_se"], marker="o")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Mean SE")
    plt.title("Unfolded WMMSE-lite SE vs Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "se_vs_latency.png")
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = _load_yaml(Path("configs/synthetic_unfolded_wmmse_lite_iter2.yaml"))

    rows = _existing_seed_variant_rows()
    for init_method, num_layers, distill_weight, delta_weight in _candidate_space(args.quick):
        tag = _variant_tag(init_method, num_layers, distill_weight, delta_weight)
        cfg_path = out_dir / f"{tag}.yaml"
        run_dir = out_dir / tag
        eval_dir = out_dir / f"{tag}_eval"
        latency_dir = out_dir / f"{tag}_latency"

        if not cfg_path.exists():
            cfg = _build_variant_config(
                base_cfg,
                init_method=init_method,
                num_layers=num_layers,
                distill_weight=distill_weight,
                delta_norm_weight=delta_weight,
                quick=args.quick,
            )
            _dump_yaml(cfg_path, cfg)
        _maybe_train_variant(cfg_path, run_dir, args.data, args.device)
        summary, curve = _evaluate_variant(cfg_path, run_dir, eval_dir, args.data, args.device)
        train_summary = _load_yaml(run_dir / "train_summary.yaml")
        latency_table = _latency_variant(cfg_path, run_dir, latency_dir, args.data, args.device)
        rows.append(
            {
                "tag": tag,
                "config_path": str(cfg_path),
                "run_dir": str(run_dir),
                "eval_dir": str(eval_dir),
                "latency_dir": str(latency_dir),
                "source": "quick" if args.quick else "full",
                "init_method": init_method,
                "num_layers": num_layers,
                "distill_weight": distill_weight,
                "delta_norm_weight": delta_weight,
                "mean_se": summary["mean_se"],
                "gap_to_wmmse": summary["mean_gap_to_best_baseline"],
                "gap_to_wmmse_iter_5": _compute_gap_to_method(curve, "unfolded_wmmse_lite", "wmmse_iter_5"),
                "latency_ms": float(latency_table["inference_latency_ms"].iloc[0]),
                "num_params": train_summary.get("num_params"),
                "train_time": train_summary.get("train_time_sec"),
            }
        )

    table = pd.DataFrame(rows)
    _write_outputs(out_dir, table)
    print(f"Saved sweep outputs to {out_dir}")


if __name__ == "__main__":
    main()
