"""Shared model factory for training and evaluation entrypoints."""

from __future__ import annotations

import torch

from beamforming.models.cnn_beamformer import CNNBeamformer
from beamforming.models.mlp_beamformer import MLPBeamformer
from beamforming.models.residual_beamformer import ResidualRZFBeamformer
from beamforming.models.unfolded_pga import UnfoldedPGABeamformer
from beamforming.models.unfolded_rzf import UnfoldedRZFBeamformer


def build_model(model_cfg: dict, data_cfg: dict) -> torch.nn.Module:
    """Instantiate a supported model from config and dataset metadata."""
    name = model_cfg["name"]
    common = {
        "num_users": int(data_cfg["num_users"]),
        "num_bs_ant": int(data_cfg["num_bs_ant"]),
        "num_rf_chains": int(data_cfg["num_rf_chains"]),
    }
    if name == "mlp":
        return MLPBeamformer(
            hybrid=bool(model_cfg.get("hybrid", False)),
            hidden_dims=model_cfg.get("hidden_dims"),
            condition_on_snr=bool(model_cfg.get("condition_on_snr", False)),
            snr_embed_dim=int(model_cfg.get("snr_embed_dim", 16)),
            residual_to_mrt=bool(model_cfg.get("residual_to_mrt", True)),
            **common,
        )
    if name == "cnn":
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
    if name == "unfolded_pga":
        return UnfoldedPGABeamformer(num_layers=int(model_cfg.get("num_layers", 3)), **common)
    if name == "unfolded_rzf":
        return UnfoldedRZFBeamformer(
            num_layers=int(model_cfg.get("num_layers", 3)),
            alpha_init=float(model_cfg.get("alpha_init", 0.05)),
            **common,
        )
    if name in {"residual_rzf", "residual_wmmse"}:
        return ResidualRZFBeamformer(
            base_method=str(model_cfg.get("base_method", "wmmse" if name == "residual_wmmse" else "rzf")),
            condition_on_snr=bool(model_cfg.get("condition_on_snr", True)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            pool_factor=int(model_cfg.get("pool_factor", 2)),
            hidden_dims=model_cfg.get("hidden_dims"),
            learnable_alpha=bool(model_cfg.get("learnable_alpha", True)),
            alpha_init=float(model_cfg.get("alpha_init", 0.1)),
            **common,
        )
    if name == "unfolded_wmmse_lite":
        from beamforming.models.unfolded_wmmse_lite import UnfoldedWMMSELiteBeamformer

        return UnfoldedWMMSELiteBeamformer(
            num_layers=int(model_cfg.get("num_layers", 3)),
            alpha_init=float(model_cfg.get("alpha_init", 0.05)),
            init_method=str(model_cfg.get("init_method", "rzf")),
            init_wmmse_iters=int(model_cfg.get("init_wmmse_iters", 2)),
            **common,
        )
    raise ValueError(f"Unsupported model name: {name}")
