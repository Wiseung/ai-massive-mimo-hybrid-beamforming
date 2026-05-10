#!/usr/bin/env python
"""Optional minimal Sionna smoke demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info = collect_sionna_env_info()
    summary = {
        "python_version": info["python_version"],
        "torch_version": info["torch_version"],
        "cuda_available": info["cuda_available"],
        "gpu_name": info["gpu_name"],
        "sionna_import_ok": info["sionna_import_ok"],
        "sionna_version": info["sionna_version"],
        "demo_status": "skipped",
        "notes": [],
    }

    if not info["sionna_import_ok"]:
        summary["notes"] = [
            "Sionna is not installed in the current environment.",
            info["install_hint"],
            "This smoke demo is optional and does not affect the v0.1.0 benchmark path.",
        ]
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved Sionna smoke summary to {out_path}")
        print(info["install_hint"])
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_users = 2
    num_bs_ant = 8
    channel = torch.randn(batch_size, num_users, num_bs_ant, dtype=torch.complex64, device=device)
    channel = (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0, device=device))
    avg_power = float(channel.abs().pow(2).mean().item())

    summary["demo_status"] = "ok"
    summary["channel_shape"] = list(channel.shape)
    summary["channel_dtype"] = str(channel.dtype)
    summary["channel_device"] = str(channel.device)
    summary["mean_channel_power"] = avg_power
    summary["notes"] = [
        "Sionna import succeeded.",
        "This is an environment smoke test only, not a full Sionna end-to-end link.",
        "No ray tracing, no RT, and no 5G NR full-stack components are exercised here.",
    ]
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Sionna smoke summary to {out_path}")


if __name__ == "__main__":
    main()
