#!/usr/bin/env python
"""Evaluate a trained beamformer checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset
from beamforming.data.deepmimo_loader import load_deepmimo_dataset
from beamforming.models.cnn_beamformer import CNNBeamformer
from beamforming.models.mlp_beamformer import MLPBeamformer
from beamforming.models.unfolded_pga import UnfoldedPGABeamformer
from beamforming.training.losses import beamforming_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--bs-idx", type=int, default=0)
    parser.add_argument("--deepmimo-users", type=int, default=4)
    parser.add_argument("--subcarrier-idx", type=int, default=None)
    return parser.parse_args()


def _build_model(model_cfg: dict, data_cfg: dict) -> torch.nn.Module:
    common = {
        "num_users": int(data_cfg["num_users"]),
        "num_bs_ant": int(data_cfg["num_bs_ant"]),
        "num_rf_chains": int(data_cfg["num_rf_chains"]),
    }
    if model_cfg["name"] == "mlp":
        return MLPBeamformer(hybrid=bool(model_cfg.get("hybrid", False)), hidden_dims=model_cfg.get("hidden_dims"), **common)
    if model_cfg["name"] == "cnn":
        return CNNBeamformer(hybrid=bool(model_cfg.get("hybrid", False)), **common)
    if model_cfg["name"] == "unfolded_pga":
        return UnfoldedPGABeamformer(num_layers=int(model_cfg.get("num_layers", 3)), **common)
    raise ValueError(f"Unsupported model: {model_cfg['name']}")


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.dataset_type == "deepmimo":
        dataset = load_deepmimo_dataset(
            args.data,
            bs_idx=args.bs_idx,
            num_users=args.deepmimo_users,
            subcarrier_idx=args.subcarrier_idx,
        )
    else:
        dataset = load_channel_dataset(args.data)
    data_cfg = {**dataset.metadata}
    data_cfg.setdefault("num_users", dataset.channels.shape[-2])
    data_cfg.setdefault("num_bs_ant", dataset.channels.shape[-1])
    data_cfg.setdefault("num_rf_chains", min(data_cfg["num_users"], 4))
    model = _build_model(config["model"], data_cfg)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    loader = DataLoader(dataset, batch_size=config["evaluation"].get("batch_size", 256), shuffle=False)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            channel = batch["channel"].to(device)
            channel_real = batch["channel_real"].to(device)
            snr_db = batch["snr_db"].to(device)
            outputs = model(channel_real)
            _, stats = beamforming_loss(
                channel,
                outputs,
                snr_db,
                lambda_power=config["training"]["lambda_power"],
                lambda_const=config["training"]["lambda_const"],
            )
            rows.append(
                {
                    "batch_idx": batch_idx,
                    "sum_rate": float(stats["sum_rate"].item()),
                    "loss": float(stats["loss"].item()),
                }
            )
    csv_path = out_dir / "evaluation.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["batch_idx", "sum_rate", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mean_sum_rate": float(sum(row["sum_rate"] for row in rows) / max(len(rows), 1)),
        "mean_loss": float(sum(row["loss"] for row in rows) / max(len(rows), 1)),
    }
    with open(out_dir / "summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle)
    print(summary)


if __name__ == "__main__":
    main()
