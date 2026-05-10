#!/usr/bin/env python
"""Benchmark inference latency with a unified protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.baselines.common import get_digital_precoder, parse_wmmse_iterations
from beamforming.baselines.dft_codebook import dft_hybrid_precoder
from beamforming.data.dataset import load_channel_dataset
from beamforming.data.deepmimo_loader import load_deepmimo_dataset
from beamforming.data.splits import load_dataset_split
from beamforming.evaluation import get_eval_subset, get_eval_subset_from_payload
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr
from beamforming.models.factory import build_model
from beamforming.models.constraints import power_normalization


LEARNED_METHODS = {"cnn", "mlp", "residual_rzf", "residual_wmmse", "unfolded_rzf", "unfolded_wmmse_lite"}
UNFOLDED_SWEEP_BEST = Path("outputs/comparisons/unfolded_wmmse_lite_sweep/best_variant.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--num-subcarriers", type=int, default=None)
    parser.add_argument("--narrowband", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--timed-runs", type=int, default=100)
    parser.add_argument("--include-data-transfer", choices=["false", "true"], default="false")
    parser.add_argument("--profile-method", default=None)
    parser.add_argument(
        "--artifact-spec",
        action="append",
        default=[],
        help="Optional learned artifact override: label=model_name,config_path,ckpt_path",
    )
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace):
    if args.dataset_type == "deepmimo" or args.scenario is not None:
        return load_deepmimo_dataset(
            scenario_path=args.data,
            scenario=args.scenario,
            download=args.download,
            num_users=args.num_users,
            num_subcarriers=args.num_subcarriers,
            narrowband=args.narrowband or args.num_subcarriers in (None, 1),
        )
    if args.data is None:
        raise ValueError("--data is required for non-DeepMIMO latency benchmarking.")
    return load_channel_dataset(args.data)


def _dataset_prefix(args: argparse.Namespace) -> str:
    return "deepmimo" if args.dataset_type == "deepmimo" or args.scenario is not None else "synthetic"


def _default_artifacts(prefix: str) -> dict[str, dict[str, str]]:
    if prefix == "deepmimo":
        return {
            "cnn": {
                "model_name": "cnn",
                "config": "configs/deepmimo_cnn_finetune.yaml",
                "ckpt": "outputs/runs/deepmimo_cnn_finetune/best.pt",
            },
            "residual_rzf": {
                "model_name": "residual_rzf",
                "config": "configs/deepmimo_residual_rzf.yaml",
                "ckpt": "outputs/runs/deepmimo_residual_rzf/best.pt",
            },
            "residual_wmmse": {
                "model_name": "residual_wmmse",
                "config": "configs/deepmimo_residual_wmmse.yaml",
                "ckpt": "outputs/runs/deepmimo_residual_wmmse/best.pt",
            },
            "unfolded_rzf": {
                "model_name": "unfolded_rzf",
                "config": "configs/deepmimo_unfolded_rzf.yaml",
                "ckpt": "outputs/runs/deepmimo_unfolded_rzf/best.pt",
            },
            "unfolded_wmmse_lite": {
                "model_name": "unfolded_wmmse_lite",
                "config": "configs/deepmimo_unfolded_wmmse_lite.yaml",
                "ckpt": "outputs/runs/deepmimo_unfolded_wmmse_lite/best.pt",
            },
        }
    return {
        "cnn": {
            "model_name": "cnn",
            "config": "configs/synthetic_cnn_finetune.yaml",
            "ckpt": "outputs/runs/cnn_finetune_rzf/best.pt",
        },
        "residual_rzf": {
            "model_name": "residual_rzf",
            "config": "configs/synthetic_residual_rzf.yaml",
            "ckpt": "outputs/runs/synthetic_residual_rzf/best.pt",
        },
        "residual_wmmse": {
            "model_name": "residual_wmmse",
            "config": "configs/synthetic_residual_wmmse.yaml",
            "ckpt": "outputs/runs/synthetic_residual_wmmse_finetune/best.pt",
        },
        "unfolded_rzf": {
            "model_name": "unfolded_rzf",
            "config": "configs/synthetic_unfolded_rzf.yaml",
            "ckpt": "outputs/runs/synthetic_unfolded_rzf/best.pt",
        },
        "unfolded_wmmse_lite": {
            "model_name": "unfolded_wmmse_lite",
            "config": "configs/synthetic_unfolded_wmmse_lite_iter2.yaml",
            "ckpt": "outputs/runs/synthetic_unfolded_wmmse_lite_iter2/best.pt",
        },
    }


def _maybe_override_best_unfolded(prefix: str, artifact_map: dict[str, dict[str, str]]) -> None:
    if prefix != "synthetic" or not UNFOLDED_SWEEP_BEST.exists():
        return
    payload = yaml.safe_load(UNFOLDED_SWEEP_BEST.read_text(encoding="utf-8")) or {}
    best_variant = payload.get("best_by_se", {})
    config_path = str(best_variant.get("config_path", "")).strip()
    run_dir = str(best_variant.get("run_dir", "")).strip()
    ckpt_path = str(Path(run_dir) / "best.pt") if run_dir else ""
    if config_path and ckpt_path and Path(config_path).exists() and Path(ckpt_path).exists():
        artifact_map["unfolded_wmmse_lite"] = {
            "model_name": "unfolded_wmmse_lite",
            "config": config_path,
            "ckpt": ckpt_path,
        }


def _parse_artifact_specs(specs: list[str]) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid artifact spec: {spec}")
        label, payload = spec.split("=", 1)
        parts = [part.strip() for part in payload.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid artifact spec: {spec}")
        model_name, config_path, ckpt_path = parts
        parsed[label.strip()] = {"model_name": model_name, "config": config_path, "ckpt": ckpt_path}
    return parsed


def _reference_config(prefix: str) -> Path:
    return Path("configs/deepmimo_residual_rzf.yaml" if prefix == "deepmimo" else "configs/synthetic_residual_rzf.yaml")


def _select_eval_subset(dataset, split_path: str | None, prefix: str):
    if split_path:
        split_payload = load_dataset_split(split_path)
        return get_eval_subset_from_payload(dataset, split_payload)
    cfg = yaml.safe_load(_reference_config(prefix).read_text(encoding="utf-8"))
    return get_eval_subset(dataset, val_fraction=float(cfg["training"].get("val_fraction", 0.1)), seed=int(cfg["training"]["seed"]))


def _get_batch(eval_subset, batch_size: int) -> dict[str, torch.Tensor]:
    loader = DataLoader(eval_subset, batch_size=min(batch_size, len(eval_subset)), shuffle=False)
    batch = next(iter(loader))
    return batch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_learned_model(
    label: str,
    artifact: dict[str, str],
    data_cfg: dict,
    device: torch.device,
) -> torch.nn.Module:
    config_path = Path(artifact["config"])
    ckpt_path = Path(artifact["ckpt"])
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing learned artifact for {label}: {config_path}, {ckpt_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "model_name" in artifact:
        config = {**config, "model": {**config["model"], "name": artifact["model_name"]}}
    model = build_model(config["model"], data_cfg)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)
    model.eval()
    return model


def _prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _baseline_forward(method: str, batch: dict[str, torch.Tensor], num_rf_chains: int) -> None:
    channel = batch["channel"]
    noise_var = batch["noise_var"]
    if method in {"mrt", "zf", "rzf", "wmmse"} or parse_wmmse_iterations(method) is not None or method in {"fd_zf", "fd_rzf"}:
        get_digital_precoder(method, channel, noise_var=noise_var)
        return
    if method == "dft":
        dft_hybrid_precoder(channel, num_rf_chains=num_rf_chains)
        return
    raise ValueError(f"Unsupported latency method: {method}")


def _learned_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        model(batch["channel_real"], snr_db=batch["snr_db"], channel_complex=batch["channel"])


def _profile_unfolded_wmmse_lite(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, object]:
    if not hasattr(model, "_init_precoder") or not hasattr(model, "_steps"):
        raise ValueError("Profiling is only implemented for unfolded_wmmse_lite.")

    channel_real = batch["channel_real"]
    channel_complex = batch["channel"]
    snr_db = batch["snr_db"]

    for _ in range(warmup_runs):
        with torch.no_grad():
            model(channel_real, snr_db=snr_db, channel_complex=channel_complex)
        _sync(device)

    noise_times: list[float] = []
    init_times: list[float] = []
    refine_times: list[float] = []
    norm_times: list[float] = []
    total_times: list[float] = []

    for _ in range(timed_runs):
        _sync(device)
        t0 = time.perf_counter()
        noise_var = noise_variance_from_snr(snr_db).to(channel_real.device)
        _sync(device)
        t1 = time.perf_counter()
        precoder = model._init_precoder(channel_complex, noise_var=noise_var)
        _sync(device)
        t2 = time.perf_counter()
        norm_acc = 0.0
        refine_start = time.perf_counter()
        for step in model._steps():
            with torch.enable_grad():
                work = precoder.detach().clone().requires_grad_(True)
                rate = multi_user_downlink_sum_rate(channel_complex, work, noise_var)
                grad = torch.autograd.grad(rate.mean(), work, retain_graph=False, create_graph=False)[0]
            t_norm0 = time.perf_counter()
            precoder = power_normalization(work + step.to(channel_real.device) * grad)
            _sync(device)
            t_norm1 = time.perf_counter()
            norm_acc += (t_norm1 - t_norm0) * 1000.0
        _sync(device)
        t3 = time.perf_counter()
        noise_times.append((t1 - t0) * 1000.0)
        init_times.append((t2 - t1) * 1000.0)
        refine_times.append((t3 - refine_start) * 1000.0)
        norm_times.append(norm_acc)
        total_times.append((t3 - t0) * 1000.0)

    noise_mean = float(pd.Series(noise_times, dtype="float64").mean())
    init_mean = float(pd.Series(init_times, dtype="float64").mean())
    refine_mean = float(pd.Series(refine_times, dtype="float64").mean())
    norm_mean = float(pd.Series(norm_times, dtype="float64").mean())
    total_mean = float(pd.Series(total_times, dtype="float64").mean())
    return {
        "method": "unfolded_wmmse_lite",
        "device": str(device),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "num_layers": int(getattr(model, "num_layers", 0)),
        "init_method": str(getattr(model, "init_method", "unknown")),
        "noise_var_ms": noise_mean,
        "init_computation_time_ms": init_mean,
        "layer_refinement_time_ms": refine_mean,
        "normalization_time_ms": norm_mean,
        "model_forward_time_ms": total_mean,
        "latency_hotspot": "init_computation" if init_mean >= refine_mean else "layer_refinement",
    }


def _benchmark_callable(
    fn,
    batch_cpu: dict[str, torch.Tensor],
    batch_device: dict[str, torch.Tensor],
    device: torch.device,
    warmup_runs: int,
    timed_runs: int,
    include_data_transfer: bool,
) -> tuple[float, float]:
    for _ in range(warmup_runs):
        payload = _prepare_batch(batch_cpu, device) if include_data_transfer else batch_device
        fn(payload)
        _sync(device)
    times_ms: list[float] = []
    for _ in range(timed_runs):
        payload = _prepare_batch(batch_cpu, device) if include_data_transfer else batch_device
        start = time.perf_counter()
        fn(payload)
        _sync(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    series = pd.Series(times_ms, dtype="float64")
    return float(series.mean()), float(series.std(ddof=0))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_data_transfer = args.include_data_transfer == "true"
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = _load_dataset(args)
    prefix = _dataset_prefix(args)
    eval_subset = _select_eval_subset(dataset, args.split, prefix)
    batch_cpu = _get_batch(eval_subset, args.batch_size)
    batch_device = _prepare_batch(batch_cpu, device)

    data_cfg = {**dataset.metadata}
    data_cfg.setdefault("num_users", dataset.channels.shape[-2])
    data_cfg.setdefault("num_bs_ant", dataset.channels.shape[-1])
    data_cfg.setdefault("num_rf_chains", min(data_cfg["num_users"], 4))
    num_rf_chains = int(data_cfg["num_rf_chains"])

    artifact_map = _default_artifacts(prefix)
    _maybe_override_best_unfolded(prefix, artifact_map)
    artifact_map.update(_parse_artifact_specs(args.artifact_spec))

    rows: list[dict[str, object]] = []
    profile_payload: dict[str, object] | None = None
    for method in args.methods:
        if method in LEARNED_METHODS or method in artifact_map:
            artifact = artifact_map.get(method)
            if artifact is None:
                raise ValueError(f"No learned artifact mapping for method: {method}")
            model = _load_learned_model(method, artifact, data_cfg, device)

            def _fn(payload: dict[str, torch.Tensor]) -> None:
                _learned_forward(model, payload)

            mean_ms, std_ms = _benchmark_callable(
                _fn,
                batch_cpu,
                batch_device,
                device,
                args.warmup_runs,
                args.timed_runs,
                include_data_transfer,
            )
            if args.profile_method == method:
                profile_payload = _profile_unfolded_wmmse_lite(
                    model,
                    batch_device,
                    device,
                    args.warmup_runs,
                    args.timed_runs,
                )
        else:

            def _fn(payload: dict[str, torch.Tensor]) -> None:
                _baseline_forward(method, payload, num_rf_chains)

            mean_ms, std_ms = _benchmark_callable(
                _fn,
                batch_cpu,
                batch_device,
                device,
                args.warmup_runs,
                args.timed_runs,
                include_data_transfer,
            )

        rows.append(
            {
                "method": method,
                "batch_size": int(batch_device["channel"].size(0)),
                "warmup_runs": args.warmup_runs,
                "timed_runs": args.timed_runs,
                "include_data_transfer": include_data_transfer,
                "device": str(device),
                "inference_latency_ms": mean_ms,
                "latency_std_ms": std_ms,
            }
        )

    table = pd.DataFrame(rows).sort_values("inference_latency_ms").reset_index(drop=True)
    table.to_csv(out_dir / "latency_table.csv", index=False)
    if profile_payload is not None:
        (out_dir / f"{args.profile_method}_profile.json").write_text(
            json.dumps(profile_payload, indent=2),
            encoding="utf-8",
        )

    plt.figure(figsize=(8.0, 4.8))
    plt.bar(table["method"], table["inference_latency_ms"], yerr=table["latency_std_ms"], capsize=3)
    plt.ylabel("Inference latency (ms)")
    plt.title("Latency Benchmark")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_bar.png")
    plt.close()
    print(f"Saved latency table to {out_dir / 'latency_table.csv'}")


if __name__ == "__main__":
    main()
