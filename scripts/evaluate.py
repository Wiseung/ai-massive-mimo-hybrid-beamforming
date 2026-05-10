#!/usr/bin/env python
"""Evaluate a trained beamformer checkpoint on a fair SNR benchmark."""

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
from beamforming.data.splits import load_dataset_split
from beamforming.evaluation import (
    add_relative_gaps,
    evaluate_baselines_by_snr,
    evaluate_model_by_snr,
    get_eval_subset,
    get_eval_subset_from_payload,
)
from beamforming.models.cnn_beamformer import CNNBeamformer
from beamforming.models.mlp_beamformer import MLPBeamformer
from beamforming.models.residual_beamformer import ResidualRZFBeamformer
from beamforming.models.unfolded_pga import UnfoldedPGABeamformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
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


def _build_model(model_cfg: dict, data_cfg: dict) -> torch.nn.Module:
    common = {
        "num_users": int(data_cfg["num_users"]),
        "num_bs_ant": int(data_cfg["num_bs_ant"]),
        "num_rf_chains": int(data_cfg["num_rf_chains"]),
    }
    if model_cfg["name"] == "mlp":
        return MLPBeamformer(
            hybrid=bool(model_cfg.get("hybrid", False)),
            hidden_dims=model_cfg.get("hidden_dims"),
            condition_on_snr=bool(model_cfg.get("condition_on_snr", False)),
            snr_embed_dim=int(model_cfg.get("snr_embed_dim", 16)),
            residual_to_mrt=bool(model_cfg.get("residual_to_mrt", True)),
            **common,
        )
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
    if model_cfg["name"] == "unfolded_pga":
        return UnfoldedPGABeamformer(num_layers=int(model_cfg.get("num_layers", 3)), **common)
    if model_cfg["name"] == "residual_rzf":
        return ResidualRZFBeamformer(
            condition_on_snr=bool(model_cfg.get("condition_on_snr", True)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            pool_factor=int(model_cfg.get("pool_factor", 2)),
            hidden_dims=model_cfg.get("hidden_dims"),
            learnable_alpha=bool(model_cfg.get("learnable_alpha", True)),
            alpha_init=float(model_cfg.get("alpha_init", 0.1)),
            **common,
        )
    raise ValueError(f"Unsupported model: {model_cfg['name']}")


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
        raise ValueError("--data is required for non-DeepMIMO evaluation.")
    return load_channel_dataset(args.data)


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset = _load_dataset(args)
    split_payload = load_dataset_split(args.split) if args.split else None
    eval_subset = (
        get_eval_subset_from_payload(dataset, split_payload)
        if split_payload is not None
        else get_eval_subset(
            dataset,
            val_fraction=float(config["training"].get("val_fraction", 0.1)),
            seed=int(config["training"]["seed"]),
        )
    )
    data_cfg = {**dataset.metadata}
    data_cfg.setdefault("num_users", dataset.channels.shape[-2])
    data_cfg.setdefault("num_bs_ant", dataset.channels.shape[-1])
    data_cfg.setdefault("num_rf_chains", min(data_cfg["num_users"], 4))
    model = _build_model(config["model"], data_cfg)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)

    learned_df = evaluate_model_by_snr(model, eval_subset, batch_size=config["evaluation"].get("batch_size", 256), device=device)
    learned_df["method"] = config["model"]["name"]
    baseline_df = evaluate_baselines_by_snr(["mrt", "zf", "rzf", "dft", "fd_zf", "fd_rzf"], eval_subset, num_rf_chains=int(data_cfg["num_rf_chains"]))
    combined = pd.concat([baseline_df, learned_df[["method", "snr_db", "se", "runtime_sec"]]], ignore_index=True)
    combined = add_relative_gaps(combined)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "evaluation_by_snr.csv"
    combined.to_csv(csv_path, index=False)
    summary = {
        "mean_se": float(learned_df["se"].mean()),
        "se_by_snr": {str(row.snr_db): float(row.se) for row in learned_df.itertuples()},
        "mean_relative_gap_to_rzf": float(
            combined[combined["method"] == config["model"]["name"]]["relative_gap_to_rzf"].mean()
        ),
        "mean_relative_gap_to_best_baseline": float(
            combined[combined["method"] == config["model"]["name"]]["relative_gap_to_best_baseline"].mean()
        ),
        "mean_gap_to_strongest_reference": float(
            combined[combined["method"] == config["model"]["name"]]["relative_gap_to_strongest_reference"].mean()
        ),
        "gap_10db": float(
            combined[(combined["method"] == config["model"]["name"]) & (combined["snr_db"] == 10.0)]["relative_gap_to_rzf"].iloc[0]
        ),
        "gap_15db": float(
            combined[(combined["method"] == config["model"]["name"]) & (combined["snr_db"] == 15.0)]["relative_gap_to_rzf"].iloc[0]
        ),
        "gap_20db": float(
            combined[(combined["method"] == config["model"]["name"]) & (combined["snr_db"] == 20.0)]["relative_gap_to_rzf"].iloc[0]
        ),
        "mean_gap_high_snr": float(
            combined[
                (combined["method"] == config["model"]["name"]) & (combined["snr_db"].isin([10.0, 15.0, 20.0]))
            ]["relative_gap_to_rzf"].mean()
        ),
    }
    with open(out_dir / "summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle)
    print(summary)


if __name__ == "__main__":
    main()
