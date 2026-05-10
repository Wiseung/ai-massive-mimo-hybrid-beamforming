"""Training loop for beamforming models."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from beamforming.data.dataset import split_dataset
from beamforming.training.losses import beamforming_loss
from beamforming.utils.seed import set_seed


@dataclass
class TrainerConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    num_workers: int
    lambda_power: float
    lambda_const: float
    amp: bool
    val_fraction: float
    snr_loss_weights: list[dict[str, float]] | None = None
    selection_metric: str = "val_sum_rate"
    selection_mode: str = "max"


def _prepare_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gradient_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().norm(2).item() ** 2)
    return total ** 0.5


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    lambda_power: float,
    lambda_const: float,
    loss_fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: dict[str, float] = {
        "loss": 0.0,
        "sum_rate": 0.0,
        "weighted_sum_rate": 0.0,
        "power_violation": 0.0,
        "constant_modulus_violation": 0.0,
        "precoder_norm": 0.0,
        "gradient_norm": 0.0,
    }
    total_batches = 0
    for batch in tqdm(loader, leave=False):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(batch["channel_real"], snr_db=batch["snr_db"], channel_complex=batch["channel"])
                loss, stats = loss_fn(batch, outputs)
            grad_norm = 0.0
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = _gradient_norm(model)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = _gradient_norm(model)
                    optimizer.step()
            for key in ("loss", "sum_rate", "weighted_sum_rate", "power_violation", "constant_modulus_violation", "precoder_norm"):
                totals[key] += float(stats[key].item())
            totals["gradient_norm"] += grad_norm
            total_batches += 1
    lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
    return {
        key: value / max(total_batches, 1)
        for key, value in totals.items()
    } | {"learning_rate": lr}


def train_model(
    model: torch.nn.Module,
    dataset,
    config: TrainerConfig,
    out_dir: str | Path,
    device: str | None = None,
    resume: str | None = None,
    init_ckpt: str | None = None,
    loss_fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, torch.Tensor]]] | None = None,
) -> dict[str, Any]:
    """Train a beamforming model and save checkpoints and logs."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    if loss_fn is None:
        def default_loss_fn(batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return beamforming_loss(
                channel=batch["channel"],
                outputs=outputs,
                snr_db=batch["snr_db"],
                lambda_power=config.lambda_power,
                lambda_const=config.lambda_const,
                snr_loss_weights={
                    float(item["snr"]): float(item["weight"]) for item in (config.snr_loss_weights or [])
                } if config.snr_loss_weights else None,
            )
        loss_fn = default_loss_fn

    device_obj = _prepare_device(device)
    amp_enabled = config.amp and device_obj.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled) if device_obj.type == "cuda" else None

    train_set, val_set = split_dataset(dataset, val_fraction=config.val_fraction, seed=config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = model.to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_epoch = 0
    selection_metric = config.selection_metric
    selection_mode = config.selection_mode.lower()
    if selection_mode not in {"max", "min"}:
        raise ValueError(f"Unsupported selection_mode: {config.selection_mode}")
    best_selection_value = float("-inf") if selection_mode == "max" else float("inf")
    best_val_sum_rate = float("-inf")
    best_epoch = -1
    train_start = time.perf_counter()
    if init_ckpt:
        init_state = torch.load(init_ckpt, map_location=device_obj, weights_only=False)
        model.load_state_dict(init_state["model"], strict=False)
    if resume:
        checkpoint = torch.load(resume, map_location=device_obj, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_selection_value = float(
            checkpoint.get("best_selection_value", checkpoint.get("best_val_sum_rate", best_selection_value))
        )
        best_val_sum_rate = float(checkpoint.get("best_val_sum_rate", best_val_sum_rate))

    writer = SummaryWriter(log_dir=str(out_path / "tensorboard"))
    csv_path = out_path / "train_log.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_sum_rate",
        "train_power_violation",
        "train_constant_modulus_violation",
        "train_precoder_norm",
        "train_weighted_sum_rate",
        "train_gradient_norm",
        "train_learning_rate",
        "val_loss",
        "val_sum_rate",
        "val_weighted_sum_rate",
        "val_power_violation",
        "val_constant_modulus_violation",
        "val_precoder_norm",
        "val_gradient_norm",
        "val_learning_rate",
    ]
    with csv_path.open("a", newline="") as csv_file:
        writer_obj = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            writer_obj.writeheader()
        for epoch in range(start_epoch, config.epochs):
            train_stats = _run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device_obj,
                amp_enabled,
                config.lambda_power,
                config.lambda_const,
                loss_fn,
            )
            val_stats = _run_epoch(
                model,
                val_loader,
                optimizer=None,
                scaler=None,
                device=device_obj,
                amp_enabled=amp_enabled,
                lambda_power=config.lambda_power,
                lambda_const=config.lambda_const,
                loss_fn=loss_fn,
            )
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_sum_rate": train_stats["sum_rate"],
                "train_power_violation": train_stats["power_violation"],
                "train_constant_modulus_violation": train_stats["constant_modulus_violation"],
                "train_precoder_norm": train_stats["precoder_norm"],
                "train_weighted_sum_rate": train_stats["weighted_sum_rate"],
                "train_gradient_norm": train_stats["gradient_norm"],
                "train_learning_rate": train_stats["learning_rate"],
                "val_loss": val_stats["loss"],
                "val_sum_rate": val_stats["sum_rate"],
                "val_weighted_sum_rate": val_stats["weighted_sum_rate"],
                "val_power_violation": val_stats["power_violation"],
                "val_constant_modulus_violation": val_stats["constant_modulus_violation"],
                "val_precoder_norm": val_stats["precoder_norm"],
                "val_gradient_norm": val_stats["gradient_norm"],
                "val_learning_rate": val_stats["learning_rate"],
            }
            writer_obj.writerow(row)
            csv_file.flush()
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_sum_rate": best_val_sum_rate,
                "best_selection_value": best_selection_value,
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "config": config.__dict__,
            }
            torch.save(checkpoint, out_path / "last.pt")
            current_selection_value = row[selection_metric]
            improved = (
                current_selection_value > best_selection_value
                if selection_mode == "max"
                else current_selection_value < best_selection_value
            )
            if improved:
                best_selection_value = current_selection_value
                best_val_sum_rate = row["val_sum_rate"]
                best_epoch = epoch
                checkpoint["best_val_sum_rate"] = best_val_sum_rate
                checkpoint["best_selection_value"] = best_selection_value
                torch.save(checkpoint, out_path / "best.pt")
    writer.close()
    train_time_sec = time.perf_counter() - train_start
    return {
        "best_val_sum_rate": best_val_sum_rate,
        "best_selection_value": best_selection_value,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "best_epoch": best_epoch,
        "train_time_sec": train_time_sec,
        "out_dir": str(out_path),
        "device": str(device_obj),
    }
