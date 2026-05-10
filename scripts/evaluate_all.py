#!/usr/bin/env python
"""Unified fair evaluation for baselines and learned beamformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset
from beamforming.data.deepmimo_loader import load_deepmimo_dataset
from beamforming.evaluation import add_relative_gaps, evaluate_baselines_by_snr, evaluate_model_by_snr, get_eval_subset, save_comparison_outputs
from beamforming.models.cnn_beamformer import CNNBeamformer
from beamforming.models.mlp_beamformer import MLPBeamformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--num-subcarriers", type=int, default=None)
    parser.add_argument("--narrowband", action="store_true")
    parser.add_argument("--device", default="auto")
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
        raise ValueError("--data is required for non-DeepMIMO evaluation.")
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
    raise ValueError(f"Unsupported learned model for evaluate_all: {model_cfg['name']}")


def main() -> None:
    args = parse_args()
    dataset = _load_dataset(args)
    methods = args.methods
    learned_methods = [m for m in methods if m in {"cnn", "mlp"}]
    baseline_methods = [m for m in methods if m not in {"cnn", "mlp"}]
    combined_frames: list[pd.DataFrame] = []

    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        eval_subset = get_eval_subset(
            dataset,
            val_fraction=float(config["training"].get("val_fraction", 0.1)),
            seed=int(config["training"]["seed"]),
        )
        data_cfg = {**dataset.metadata}
        data_cfg.setdefault("num_users", dataset.channels.shape[-2])
        data_cfg.setdefault("num_bs_ant", dataset.channels.shape[-1])
        data_cfg.setdefault("num_rf_chains", min(data_cfg["num_users"], 4))
    else:
        config = None
        eval_subset = dataset
        data_cfg = {**dataset.metadata}
        data_cfg.setdefault("num_users", dataset.channels.shape[-2])
        data_cfg.setdefault("num_bs_ant", dataset.channels.shape[-1])
        data_cfg.setdefault("num_rf_chains", min(data_cfg["num_users"], 4))

    if baseline_methods:
        combined_frames.append(evaluate_baselines_by_snr(baseline_methods, eval_subset, num_rf_chains=int(data_cfg["num_rf_chains"])))

    if learned_methods:
        if args.ckpt is None or config is None:
            raise ValueError("--ckpt and --config are required when evaluating learned methods.")
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = _build_model(config["model"], data_cfg)
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        model = model.to(device)
        learned_df = evaluate_model_by_snr(model, eval_subset, batch_size=config["evaluation"].get("batch_size", 256), device=device)
        learned_df["method"] = config["model"]["name"]
        if config["model"]["name"] in learned_methods:
            combined_frames.append(learned_df[["method", "snr_db", "se", "runtime_sec"]])

    combined = pd.concat(combined_frames, ignore_index=True)
    combined = add_relative_gaps(combined)
    prefix = "synthetic" if args.dataset_type != "deepmimo" and args.scenario is None else "deepmimo"
    csv_path, fig_path = save_comparison_outputs(combined, args.out, prefix=prefix)
    print(f"Saved unified comparison CSV to {csv_path}")
    print(f"Saved unified comparison figure to {fig_path}")


if __name__ == "__main__":
    main()
