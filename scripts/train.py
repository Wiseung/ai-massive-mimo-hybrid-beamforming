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
from beamforming.data.splits import load_dataset_split
from beamforming.models.factory import build_model
from beamforming.models.unfolded_pga import UnfoldedPGABeamformer
from beamforming.training.losses import beamforming_loss
from beamforming.training.supervised_targets import get_teacher_target
from beamforming.training.trainer import TrainerConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--bs-idx", type=int, default=0)
    parser.add_argument("--deepmimo-users", type=int, default=4)
    parser.add_argument("--subcarrier-idx", type=int, default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-subcarriers", type=int, default=None)
    parser.add_argument("--narrowband", action="store_true")
    return parser.parse_args()

def _load_dataset(args: argparse.Namespace):
    if args.dataset_type == "deepmimo" or args.scenario is not None:
        return load_deepmimo_dataset(
            scenario_path=args.data,
            scenario=args.scenario,
            download=args.download,
            bs_idx=args.bs_idx,
            num_users=args.deepmimo_users,
            subcarrier_idx=args.subcarrier_idx,
            num_subcarriers=args.num_subcarriers,
            narrowband=args.narrowband or args.num_subcarriers in (None, 1),
        )
    if args.data is None:
        raise ValueError("--data is required for non-DeepMIMO training.")
    return load_channel_dataset(args.data)


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset = _load_dataset(args)
    data_cfg = {**dataset.metadata}
    data_cfg.setdefault("num_users", dataset.channels.shape[-2])
    data_cfg.setdefault("num_bs_ant", dataset.channels.shape[-1])
    data_cfg.setdefault("num_rf_chains", min(data_cfg["num_users"], 4))
    model = build_model(config["model"], data_cfg)
    trainer_field_names = {field.name for field in fields(TrainerConfig)}
    trainer_kwargs = {key: value for key, value in config["training"].items() if key in trainer_field_names}
    trainer_cfg = TrainerConfig(**trainer_kwargs)
    split_payload = load_dataset_split(args.split) if args.split else None
    loss_cfg = config.get("loss", {})

    loss_fn = None
    if loss_cfg:
        rate_weight = float(loss_cfg.get("rate_weight", 1.0))
        distill_weight = float(loss_cfg.get("distill_weight", 0.0))
        distill_teacher = loss_cfg.get("teacher")
        teacher_max_iter = int(loss_cfg.get("teacher_max_iter", 30))
        lambda_delta = float(loss_cfg.get("delta_norm_weight", config["training"].get("lambda_delta", 0.0)))

        def configured_loss_fn(batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]):
            distill_target = None
            if distill_teacher:
                teacher_name = str(distill_teacher)
                if teacher_name == "wmmse" and teacher_max_iter > 0:
                    teacher_name = f"wmmse_iter_{teacher_max_iter}"
                distill_target = get_teacher_target(batch["channel"], batch["snr_db"], teacher_name)
            return beamforming_loss(
                channel=batch["channel"],
                outputs=outputs,
                snr_db=batch["snr_db"],
                lambda_power=trainer_cfg.lambda_power,
                lambda_const=trainer_cfg.lambda_const,
                snr_loss_weights={
                    float(item["snr"]): float(item["weight"]) for item in (trainer_cfg.snr_loss_weights or [])
                } if trainer_cfg.snr_loss_weights else None,
                lambda_delta=lambda_delta,
                rate_weight=rate_weight,
                distill_target=distill_target,
                distill_weight=distill_weight,
            )

        loss_fn = configured_loss_fn
    result = train_model(
        model,
        dataset,
        trainer_cfg,
        out_dir=args.out,
        device=args.device,
        resume=args.resume,
        init_ckpt=args.init_ckpt,
        split_payload=split_payload,
        loss_fn=loss_fn,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "train_summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle)
    print("Training complete.")
    print(result)


if __name__ == "__main__":
    main()
