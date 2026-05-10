#!/usr/bin/env python
"""Run a short reproducibility smoke check and export a compact summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.baselines.common import evaluate_baseline
from beamforming.data.dataset import load_channel_dataset
from beamforming.data.synthetic import sparse_geometric_mmwave_channel
from beamforming.evaluation import evaluate_model_by_snr
from beamforming.models.factory import build_model


DEFAULT_DATA = Path("outputs/data/synthetic_narrowband.pt")
DEFAULT_MODEL_CONFIG = Path("configs/synthetic_residual_rzf.yaml")
DEFAULT_MODEL_CKPT = Path("outputs/runs/synthetic_residual_rzf/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _ensure_dataset(path: Path) -> tuple[Path, bool]:
    if path.exists():
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = sparse_geometric_mmwave_channel(
        num_samples=256,
        num_users=4,
        num_bs_ant=64,
        num_paths=3,
    )
    payload = {
        "channels": channels.cpu(),
        "snr_db": torch.tensor([-10, -5, 0, 5, 10, 15, 20], dtype=torch.float32),
        "metadata": {
            "channel_type": "mmwave",
            "num_samples": 256,
            "num_bs_ant": 64,
            "num_users": 4,
            "num_rf_chains": 4,
            "num_paths": 3,
            "seed": 42,
            "generated_by": "scripts/reproduce_minimal.py",
        },
    }
    torch.save(payload, path)
    return path, True


def _device_from_arg(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data_path, generated_data = _ensure_dataset(Path(args.data))
        dataset = load_channel_dataset(data_path)
        device = _device_from_arg(args.device)

        subset_channels = dataset.channels[: min(128, len(dataset))].to(device)
        sample_snr = 10.0
        rzf_result = evaluate_baseline("rzf", subset_channels, sample_snr, num_rf_chains=4)
        wmmse_iter_5_result = evaluate_baseline("wmmse_iter_5", subset_channels, sample_snr, num_rf_chains=4)

        model_summary: dict[str, object]
        if DEFAULT_MODEL_CONFIG.exists() and DEFAULT_MODEL_CKPT.exists():
            config = yaml.safe_load(DEFAULT_MODEL_CONFIG.read_text(encoding="utf-8"))
            data_cfg = {**dataset.metadata}
            data_cfg.setdefault("num_users", dataset.channels.shape[-2])
            data_cfg.setdefault("num_bs_ant", dataset.channels.shape[-1])
            data_cfg.setdefault("num_rf_chains", min(data_cfg["num_users"], 4))
            model = build_model(config["model"], data_cfg)
            checkpoint = torch.load(DEFAULT_MODEL_CKPT, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=False)
            model = model.to(device)
            eval_df = evaluate_model_by_snr(
                model,
                dataset,
                batch_size=min(128, len(dataset)),
                device=device,
            )
            model_summary = {
                "status": "evaluated",
                "method": config["model"]["name"],
                "config": str(DEFAULT_MODEL_CONFIG),
                "checkpoint": str(DEFAULT_MODEL_CKPT),
                "mean_se": float(eval_df["se"].mean()),
            }
        else:
            model_summary = {
                "status": "skipped",
                "reason": "default residual_rzf checkpoint not found",
                "expected_config": str(DEFAULT_MODEL_CONFIG),
                "expected_checkpoint": str(DEFAULT_MODEL_CKPT),
            }

        payload = {
            "status": "ok",
            "device": str(device),
            "data_path": str(data_path),
            "generated_data": generated_data,
            "num_samples_used": int(subset_channels.size(0)),
            "synthetic_dataset_shape": list(dataset.channels.shape),
            "baseline_summary": {
                "snr_db": sample_snr,
                "rzf_mean_se": float(rzf_result["sum_rate"].mean().item()),
                "wmmse_iter_5_mean_se": float(wmmse_iter_5_result["sum_rate"].mean().item()),
            },
            "model_summary": model_summary,
            "notes": [
                "This is a short reproducibility smoke check, not a full experiment rerun.",
                "The script never downloads DeepMIMO and does not retrain large models.",
            ],
        }
    except Exception as exc:  # pragma: no cover - script mode
        payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hint": "Check whether the synthetic dataset and optional default checkpoint exist, then rerun the script.",
        }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved minimal reproducibility summary to {out_path}")
    if payload.get("status") != "ok":
        print(f"{payload['error_type']}: {payload['error']}")


if __name__ == "__main__":
    main()
