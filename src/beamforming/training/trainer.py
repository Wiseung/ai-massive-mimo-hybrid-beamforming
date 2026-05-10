"""Training loop for beamforming models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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


def _prepare_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    lambda_power: float,
    lambda_const: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_sum_rate = 0.0
    total_batches = 0
    for batch in tqdm(loader, leave=False):
        channel = batch["channel"].to(device)
        channel_real = batch["channel_real"].to(device)
        snr_db = batch["snr_db"].to(device)
        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(channel_real)
                loss, stats = beamforming_loss(
                    channel=channel,
                    outputs=outputs,
                    snr_db=snr_db,
                    lambda_power=lambda_power,
                    lambda_const=lambda_const,
                )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(stats["loss"].item())
        total_sum_rate += float(stats["sum_rate"].item())
        total_batches += 1
    return {
        "loss": total_loss / max(total_batches, 1),
        "sum_rate": total_sum_rate / max(total_batches, 1),
    }


def train_model(
    model: torch.nn.Module,
    dataset,
    config: TrainerConfig,
    out_dir: str | Path,
    device: str | None = None,
    resume: str | None = None,
) -> dict[str, Any]:
    """Train a beamforming model and save checkpoints and logs."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    device_obj = _prepare_device(device)
    amp_enabled = config.amp and device_obj.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled) if device_obj.type == "cuda" else None

    total_len = len(dataset)
    val_len = max(1, int(total_len * config.val_fraction))
    train_len = total_len - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = model.to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_epoch = 0
    best_val = float("-inf")
    if resume:
        checkpoint = torch.load(resume, map_location=device_obj, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val_sum_rate"])

    writer = SummaryWriter(log_dir=str(out_path / "tensorboard"))
    csv_path = out_path / "train_log.csv"
    with csv_path.open("a", newline="") as csv_file:
        fieldnames = ["epoch", "train_loss", "train_sum_rate", "val_loss", "val_sum_rate"]
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
            )
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_sum_rate": train_stats["sum_rate"],
                "val_loss": val_stats["loss"],
                "val_sum_rate": val_stats["sum_rate"],
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
                "best_val_sum_rate": best_val,
                "config": config.__dict__,
            }
            torch.save(checkpoint, out_path / "last.pt")
            if val_stats["sum_rate"] > best_val:
                best_val = val_stats["sum_rate"]
                checkpoint["best_val_sum_rate"] = best_val
                torch.save(checkpoint, out_path / "best.pt")
    writer.close()
    return {"best_val_sum_rate": best_val, "out_dir": str(out_path), "device": str(device_obj)}
