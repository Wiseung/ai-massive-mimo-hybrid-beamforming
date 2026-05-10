#!/usr/bin/env python
"""Supervised pretraining for learned beamformers using classical teachers."""

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
from beamforming.training.supervised_targets import get_teacher_target
from beamforming.training.trainer import TrainerConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--teacher", choices=["mrt", "zf", "rzf", "mixed_rzf_zf", "best_of_rzf_zf"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace):
    if args.dataset_type == "deepmimo" or args.scenario is not None:
        return load_deepmimo_dataset(
            scenario_path=args.data,
            scenario=args.scenario,
            download=args.download,
            num_users=args.num_users,
            narrowband=True,
        )
    if args.data is None:
        raise ValueError("--data is required for non-DeepMIMO pretraining.")
    return load_channel_dataset(args.data)


def _build_model(model_cfg: dict, data_cfg: dict) -> torch.nn.Module:
    common = {
        "num_users": int(data_cfg["num_users"]),
        "num_bs_ant": int(data_cfg["num_bs_ant"]),
        "num_rf_chains": int(data_cfg["num_rf_chains"]),
    }
    if model_cfg["name"] == "cnn":
        return CNNBeamformer(
            hybrid=bool(model_cfg.get("hybrid", False)),
            condition_on_snr=bool(model_cfg.get("condition_on_snr", False)),
            snr_embed_dim=int(model_cfg.get("snr_embed_dim", 16)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            pool_factor=int(model_cfg.get("pool_factor", 2)),
            hidden_dims=model_cfg.get("hidden_dims"),
            residual_to_mrt=bool(model_cfg.get("residual_to_mrt", True)),
            **common,
        )
    if model_cfg["name"] == "mlp":
        return MLPBeamformer(
            hybrid=bool(model_cfg.get("hybrid", False)),
            hidden_dims=model_cfg.get("hidden_dims"),
            condition_on_snr=bool(model_cfg.get("condition_on_snr", False)),
            snr_embed_dim=int(model_cfg.get("snr_embed_dim", 16)),
            residual_to_mrt=bool(model_cfg.get("residual_to_mrt", True)),
            **common,
        )
    raise ValueError(f"Unsupported model for pretraining: {model_cfg['name']}")


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset = _load_dataset(args)
    data_cfg = {**dataset.metadata}
    data_cfg.setdefault("num_users", dataset.channels.shape[-2])
    data_cfg.setdefault("num_bs_ant", dataset.channels.shape[-1])
    data_cfg.setdefault("num_rf_chains", min(data_cfg["num_users"], 4))
    model = _build_model(config["model"], data_cfg)
    trainer_field_names = {field.name for field in fields(TrainerConfig)}
    trainer_kwargs = {key: value for key, value in config["training"].items() if key in trainer_field_names}
    trainer_cfg = TrainerConfig(**trainer_kwargs)

    def pretrain_loss(batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]):
        target = get_teacher_target(batch["channel"], batch["snr_db"], args.teacher)
        error = outputs["precoder"] - target
        mse = torch.mean(torch.abs(error) ** 2)
        precoder_power = (torch.abs(outputs["precoder"]) ** 2).sum(dim=(-2, -1))
        stats = {
            "loss": mse.detach(),
            "sum_rate": torch.tensor(0.0, device=batch["channel"].device),
            "weighted_sum_rate": torch.tensor(0.0, device=batch["channel"].device),
            "power_violation": torch.mean((precoder_power - 1.0) ** 2).detach(),
            "constant_modulus_violation": torch.tensor(0.0, device=batch["channel"].device),
            "precoder_norm": torch.mean(torch.sqrt(precoder_power.clamp_min(1e-12))).detach(),
        }
        return mse, stats

    result = train_model(
        model,
        dataset,
        trainer_cfg,
        out_dir=args.out,
        device=args.device,
        loss_fn=pretrain_loss,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "pretrain_summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump({"teacher": args.teacher, **result}, handle)
    print("Pretraining complete.")
    print(result)


if __name__ == "__main__":
    main()
