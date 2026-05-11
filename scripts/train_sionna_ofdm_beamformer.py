#!/usr/bin/env python
"""Train an optional learned beamformer on synthetic OFDM channels."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from _bootstrap import add_src_to_path
import torch
import yaml

add_src_to_path()

from beamforming.data.sionna_ofdm_synthetic import SionnaOFDMSyntheticConfig, SionnaOFDMSyntheticGenerator
from beamforming.models.factory import build_model
from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_ofdm_training import compute_link_metrics, generate_qpsk_resource_grid, simulate_multiuser_ofdm_link


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str | None, cfg_device: str | None) -> torch.device:
    device_name = requested or cfg_device or "auto"
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _apply_smoke_overrides(config: dict) -> dict:
    payload = yaml.safe_load(yaml.safe_dump(config))
    payload["training"]["epochs"] = min(int(payload["training"].get("epochs", 20)), 2)
    payload["dataset"]["batch_size"] = min(int(payload["dataset"].get("batch_size", 128)), 32)
    payload["dataset"]["num_batches_per_epoch"] = min(int(payload["dataset"].get("num_batches_per_epoch", 50)), 3)
    payload["dataset"]["num_val_batches"] = min(int(payload["dataset"].get("num_val_batches", 10)), 1)
    return payload


def _prepare_log(csv_path: Path, fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _gradient_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().norm(2).item() ** 2)
    return total ** 0.5


def _build_generator(config: dict, train: bool) -> SionnaOFDMSyntheticGenerator:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    return SionnaOFDMSyntheticGenerator(
        SionnaOFDMSyntheticConfig(
            batch_size=int(dataset_cfg["batch_size"]),
            num_subcarriers=int(dataset_cfg["num_subcarriers"]),
            num_users=int(dataset_cfg["num_users"]),
            num_bs_ant=int(dataset_cfg["num_bs_ant"]),
            channel_model=str(dataset_cfg.get("channel_model", "rayleigh")),
            sparse_mmwave_like=bool(dataset_cfg.get("sparse_mmwave_like", False)),
            num_paths=int(dataset_cfg.get("num_paths", 3)),
            snr_db_choices=list(dataset_cfg["snr_db_train"] if train else dataset_cfg.get("snr_db_eval", dataset_cfg["snr_db_train"])),
            seed=int(training_cfg["seed"]) + (0 if train else 10_000),
        )
    )


def _run_epoch(
    model: torch.nn.Module,
    generator: SionnaOFDMSyntheticGenerator,
    num_batches: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    loss_cfg: dict,
) -> tuple[dict[str, float], dict[str, object]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    notes: list[str] = []
    used_sionna_ofdm = False
    used_sionna_channel = False
    fallback_used = False
    totals = {
        "loss": 0.0,
        "mean_sum_rate": 0.0,
        "receive_mse": 0.0,
        "power_norm": 0.0,
        "power_violation": 0.0,
        "grad_norm": 0.0,
    }

    for _ in range(num_batches):
        batch = generator.sample_batch(device=device, return_symbols=False)
        context = generate_qpsk_resource_grid(
            batch_size=batch["H_f"].size(0),
            num_subcarriers=batch["H_f"].size(1),
            num_users=batch["H_f"].size(2),
            device=device,
            generator=generator.generator,
        )
        used_sionna_ofdm = used_sionna_ofdm or context.used_sionna_ofdm
        fallback_used = fallback_used or context.fallback_used
        notes.extend(context.notes)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        precoder = model(batch["H_f"], snr_db=batch["snr_db"])
        link = simulate_multiuser_ofdm_link(
            channel_f=batch["H_f"],
            precoder=precoder,
            tx_symbols=context.tx_symbols,
            noise_var=batch["noise_var"],
            snr_db=batch["snr_db"],
        )
        used_sionna_channel = used_sionna_channel or bool(link["used_sionna_channel"])
        fallback_used = fallback_used or bool(link["fallback_used"])
        if link["note"]:
            notes.append(str(link["note"]))

        metrics = compute_link_metrics(
            channel_f=batch["H_f"],
            precoder=precoder,
            tx_symbols=context.tx_symbols,
            rx_symbols=link["rx"],
            noise_var=batch["noise_var"],
        )
        loss = (
            -float(loss_cfg.get("sum_rate_weight", 1.0)) * metrics["mean_sum_rate"]
            + float(loss_cfg.get("mse_weight", 0.1)) * metrics["receive_mse"]
            + float(loss_cfg.get("power_penalty", 0.01)) * metrics["power_violation"]
        )

        grad_norm = 0.0
        if train_mode:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm = _gradient_norm(model)
            optimizer.step()

        totals["loss"] += float(loss.item())
        totals["mean_sum_rate"] += float(metrics["mean_sum_rate"].item())
        totals["receive_mse"] += float(metrics["receive_mse"].item())
        totals["power_norm"] += float(metrics["power_norm"].item())
        totals["power_violation"] += float(metrics["power_violation"].item())
        totals["grad_norm"] += grad_norm

    denom = max(num_batches, 1)
    averaged = {key: value / denom for key, value in totals.items()}
    metadata = {
        "used_sionna_ofdm": used_sionna_ofdm,
        "used_sionna_channel": used_sionna_channel,
        "fallback_used": fallback_used,
        "notes": sorted(set(notes)),
    }
    return averaged, metadata


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    if args.smoke:
        config = _apply_smoke_overrides(config)

    env_info = collect_sionna_env_info()
    if not env_info["sionna_import_ok"]:
        raise SystemExit(
            "Sionna is not installed in the current environment. Install the optional dependency with "
            "`pip install sionna-no-rt` before running the OFDM training pipeline."
        )

    device = _resolve_device(args.device, config["training"].get("device"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["training"]["seed"]))

    data_cfg = {
        "num_users": int(config["dataset"]["num_users"]),
        "num_bs_ant": int(config["dataset"]["num_bs_ant"]),
        "num_rf_chains": int(config["dataset"]["num_users"]),
    }
    model = build_model(config["model"], data_cfg).to(device)

    optimizer_name = str(config["training"].get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))

    train_generator = _build_generator(config, train=True)
    val_generator = _build_generator(config, train=False)
    num_batches_per_epoch = int(config["dataset"]["num_batches_per_epoch"])
    num_val_batches = int(config["dataset"]["num_val_batches"])
    epochs = int(config["training"]["epochs"])
    grad_clip = float(config["training"].get("grad_clip", 0.0))

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "mean_sum_rate",
        "val_mean_sum_rate",
        "mse",
        "val_mse",
        "power_norm",
        "val_power_norm",
        "power_violation",
        "val_power_violation",
        "grad_norm",
        "learning_rate",
    ]
    csv_path = out_dir / "train_log.csv"
    _prepare_log(csv_path, fieldnames)

    best_val_loss = float("inf")
    best_epoch = -1
    best_snapshot: dict[str, object] = {}
    start_time = time.perf_counter()

    for epoch in range(epochs):
        train_stats, train_meta = _run_epoch(
            model=model,
            generator=train_generator,
            num_batches=num_batches_per_epoch,
            device=device,
            optimizer=optimizer,
            grad_clip=grad_clip,
            loss_cfg=config["loss"],
        )
        val_stats, val_meta = _run_epoch(
            model=model,
            generator=val_generator,
            num_batches=num_val_batches,
            device=device,
            optimizer=None,
            grad_clip=0.0,
            loss_cfg=config["loss"],
        )
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "mean_sum_rate": train_stats["mean_sum_rate"],
            "val_mean_sum_rate": val_stats["mean_sum_rate"],
            "mse": train_stats["receive_mse"],
            "val_mse": val_stats["receive_mse"],
            "power_norm": train_stats["power_norm"],
            "val_power_norm": val_stats["power_norm"],
            "power_violation": train_stats["power_violation"],
            "val_power_violation": val_stats["power_violation"],
            "grad_norm": train_stats["grad_norm"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(row)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "config": config,
            "device": str(device),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "train_meta": train_meta,
            "val_meta": val_meta,
            "env": env_info,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            best_snapshot = {
                "train_stats": train_stats,
                "val_stats": val_stats,
                "train_meta": train_meta,
                "val_meta": val_meta,
            }
            torch.save(checkpoint, out_dir / "best.pt")

    train_time_sec = time.perf_counter() - start_time
    summary = {
        "demo_scope": "experimental_sionna_ofdm_training",
        "sionna_import_ok": env_info["sionna_import_ok"],
        "sionna_version": env_info["sionna_version"],
        "torch_version": env_info["torch_version"],
        "device": str(device),
        "smoke": bool(args.smoke),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_time_sec": train_time_sec,
        "used_sionna_ofdm": bool(best_snapshot.get("train_meta", {}).get("used_sionna_ofdm", False) or best_snapshot.get("val_meta", {}).get("used_sionna_ofdm", False)),
        "used_sionna_channel": bool(best_snapshot.get("train_meta", {}).get("used_sionna_channel", False) or best_snapshot.get("val_meta", {}).get("used_sionna_channel", False)),
        "fallback_used": bool(best_snapshot.get("train_meta", {}).get("fallback_used", False) or best_snapshot.get("val_meta", {}).get("fallback_used", False)),
        "best_train_loss": best_snapshot.get("train_stats", {}).get("loss"),
        "best_val_mean_sum_rate": best_snapshot.get("val_stats", {}).get("mean_sum_rate"),
        "best_val_mse": best_snapshot.get("val_stats", {}).get("receive_mse"),
        "notes": sorted(
            set(best_snapshot.get("train_meta", {}).get("notes", []) + best_snapshot.get("val_meta", {}).get("notes", []))
        )
        + [
            "This pipeline is an optional synthetic OFDM link-level experiment.",
            "It does not change the v0.1.0/v0.2.0 benchmark claims.",
            "It is not Sionna RT, not ray tracing, and not a 5G NR full stack.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.smoke:
        (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved training outputs to {out_dir}")


if __name__ == "__main__":
    main()
