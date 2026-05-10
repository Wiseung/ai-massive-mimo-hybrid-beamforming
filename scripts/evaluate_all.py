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
from beamforming.data.splits import load_dataset_split
from beamforming.evaluation import (
    add_relative_gaps,
    evaluate_baselines_by_snr,
    evaluate_model_by_snr,
    get_eval_subset,
    get_eval_subset_from_payload,
    save_comparison_outputs,
)
from beamforming.models.cnn_beamformer import CNNBeamformer
from beamforming.models.mlp_beamformer import MLPBeamformer
from beamforming.models.residual_beamformer import ResidualRZFBeamformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--config", default=None)
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
    raise ValueError(f"Unsupported learned model for evaluate_all: {model_cfg['name']}")


def _dataset_prefix(args: argparse.Namespace) -> str:
    return "deepmimo" if args.dataset_type == "deepmimo" or args.scenario is not None else "synthetic"


def _default_learned_artifacts(prefix: str) -> dict[str, dict[str, Path]]:
    if prefix == "deepmimo":
        return {
            "cnn": {
                "config": Path("configs/deepmimo_cnn_finetune.yaml"),
                "ckpt": Path("outputs/runs/deepmimo_cnn_finetune/best.pt"),
            },
            "residual_rzf": {
                "config": Path("configs/deepmimo_residual_rzf.yaml"),
                "ckpt": Path("outputs/runs/deepmimo_residual_rzf/best.pt"),
            },
        }
    return {
        "cnn": {
            "config": Path("configs/synthetic_cnn_finetune.yaml"),
            "ckpt": Path("outputs/runs/cnn_finetune_rzf/best.pt"),
        },
        "residual_rzf": {
            "config": Path("configs/synthetic_residual_rzf.yaml"),
            "ckpt": Path("outputs/runs/synthetic_residual_rzf/best.pt"),
        },
    }


def _evaluate_learned_method(
    method_name: str,
    config_path: Path,
    ckpt_path: Path,
    data_cfg: dict,
    eval_subset,
    device: torch.device,
) -> pd.DataFrame:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = _build_model(config["model"], data_cfg)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)
    learned_df = evaluate_model_by_snr(
        model,
        eval_subset,
        batch_size=config["evaluation"].get("batch_size", 256),
        device=device,
    )
    learned_df["method"] = method_name
    return learned_df[["method", "snr_db", "se", "runtime_sec"]]


def main() -> None:
    args = parse_args()
    dataset = _load_dataset(args)
    methods = args.methods
    learned_methods = [m for m in methods if m in {"cnn", "mlp", "residual_rzf"}]
    baseline_methods = [m for m in methods if m not in {"cnn", "mlp", "residual_rzf"}]
    combined_frames: list[pd.DataFrame] = []
    split_payload = load_dataset_split(args.split) if args.split else None

    primary_config = None
    primary_method = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        primary_config = config
        primary_method = config["model"]["name"]
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
        if args.ckpt is None or primary_config is None or primary_method is None:
            raise ValueError("--ckpt and --config are required when evaluating learned methods.")
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        default_artifacts = _default_learned_artifacts(_dataset_prefix(args))
        requested_learned = []
        seen_methods = set()
        if primary_method in learned_methods:
            requested_learned.append((primary_method, Path(args.config), Path(args.ckpt)))
            seen_methods.add(primary_method)
        for method_name in learned_methods:
            if method_name in seen_methods:
                continue
            artifact = default_artifacts.get(method_name)
            if artifact is None:
                print(f"Skipping learned method '{method_name}': no default artifact mapping is defined.")
                continue
            if not artifact["config"].exists() or not artifact["ckpt"].exists():
                print(
                    f"Skipping learned method '{method_name}': "
                    f"missing config or checkpoint ({artifact['config']}, {artifact['ckpt']})."
                )
                continue
            requested_learned.append((method_name, artifact["config"], artifact["ckpt"]))
            seen_methods.add(method_name)
        for method_name, config_path, ckpt_path in requested_learned:
            combined_frames.append(
                _evaluate_learned_method(
                    method_name=method_name,
                    config_path=config_path,
                    ckpt_path=ckpt_path,
                    data_cfg=data_cfg,
                    eval_subset=eval_subset,
                    device=device,
                )
            )

    combined = pd.concat(combined_frames, ignore_index=True)
    combined = add_relative_gaps(combined)
    prefix = _dataset_prefix(args)
    csv_path, fig_path = save_comparison_outputs(combined, args.out, prefix=prefix)
    summary = {}
    if learned_methods:
        learned_method = primary_method
        learned_df = combined[combined["method"] == learned_method]
        summary = {
            "method": learned_method,
            "mean_se": float(learned_df["se"].mean()),
            "mean_gap_to_rzf": float(learned_df["relative_gap_to_rzf"].mean()),
            "mean_gap_to_best_baseline": float(learned_df["relative_gap_to_best_baseline"].mean()),
            "mean_gap_to_strongest_reference": float(learned_df["relative_gap_to_strongest_reference"].mean()),
            "gap_10db": float(learned_df.loc[learned_df["snr_db"] == 10.0, "relative_gap_to_rzf"].iloc[0]),
            "gap_15db": float(learned_df.loc[learned_df["snr_db"] == 15.0, "relative_gap_to_rzf"].iloc[0]),
            "gap_20db": float(learned_df.loc[learned_df["snr_db"] == 20.0, "relative_gap_to_rzf"].iloc[0]),
            "mean_gap_high_snr": float(learned_df[learned_df["snr_db"].isin([10.0, 15.0, 20.0])]["relative_gap_to_rzf"].mean()),
            "evaluated_methods": sorted(combined["method"].unique().tolist()),
        }
        with open(Path(args.out) / "summary.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle)
    print(f"Saved unified comparison CSV to {csv_path}")
    print(f"Saved unified comparison figure to {fig_path}")


if __name__ == "__main__":
    main()
