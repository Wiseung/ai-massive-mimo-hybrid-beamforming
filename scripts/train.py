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
    result = train_model(
        model,
        dataset,
        trainer_cfg,
        out_dir=args.out,
        device=args.device,
        resume=args.resume,
        init_ckpt=args.init_ckpt,
        split_payload=split_payload,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "train_summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle)
    print("Training complete.")
    print(result)


if __name__ == "__main__":
    main()
