#!/usr/bin/env python
"""Benchmark inference latency with a unified protocol."""

from __future__ import annotations

import argparse
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
from beamforming.models.factory import build_model


LEARNED_METHODS = {"cnn", "mlp", "residual_rzf", "residual_wmmse", "unfolded_rzf", "unfolded_wmmse_lite"}


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
    artifact_map.update(_parse_artifact_specs(args.artifact_spec))

    rows: list[dict[str, object]] = []
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
