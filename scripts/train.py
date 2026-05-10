#!/usr/bin/env python
"""Train an AI beamformer on a saved dataset."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch
import yaml

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset
from beamforming.data.deepmimo_loader import load_deepmimo_dataset
from beamforming.models.cnn_beamformer import CNNBeamformer
from beamforming.models.mlp_beamformer import MLPBeamformer
from beamforming.models.unfolded_pga import UnfoldedPGABeamformer
from beamforming.training.trainer import TrainerConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--bs-idx", type=int, default=0)
    parser.add_argument("--deepmimo-users", type=int, default=4)
    parser.add_argument("--subcarrier-idx", type=int, default=None)
    return parser.parse_args()


def _build_model(model_cfg: dict, data_cfg: dict) -> torch.nn.Module:
    name = model_cfg["name"]
    common = {
        "num_users": int(data_cfg["num_users"]),
        "num_bs_ant": int(data_cfg["num_bs_ant"]),
        "num_rf_chains": int(data_cfg["num_rf_chains"]),
    }
    if name == "mlp":
        return MLPBeamformer(hybrid=bool(model_cfg.get("hybrid", False)), hidden_dims=model_cfg.get("hidden_dims"), **common)
    if name == "cnn":
        return CNNBeamformer(hybrid=bool(model_cfg.get("hybrid", False)), **common)
    if name == "unfolded_pga":
        return UnfoldedPGABeamformer(num_layers=int(model_cfg.get("num_layers", 3)), **common)
    raise ValueError(f"Unsupported model name: {name}")


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
    trainer_field_names = {field.name for field in fields(TrainerConfig)}
    trainer_kwargs = {key: value for key, value in config["training"].items() if key in trainer_field_names}
    trainer_cfg = TrainerConfig(**trainer_kwargs)
    result = train_model(model, dataset, trainer_cfg, out_dir=args.out, device=args.device, resume=args.resume)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "train_summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle)
    print("Training complete.")
    print(result)


if __name__ == "__main__":
    main()
