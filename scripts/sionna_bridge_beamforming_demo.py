#!/usr/bin/env python
"""Optional bridge demo between Sionna environment checks and beamforming baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.baselines.common import evaluate_baseline
from beamforming.metrics.sum_rate import noise_variance_from_snr
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
    if not info["sionna_import_ok"]:
        payload = {
            "sionna_import_ok": False,
            "bridge_status": "skipped",
            "notes": [
                "Sionna is not installed in the current environment.",
                info["install_hint"],
                "The bridge demo is intentionally optional and does not change the core benchmark path.",
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved bridge summary to {out_path}")
        print(info["install_hint"])
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_users = 4
    num_bs_ant = 16
    snr_db = 10.0
    channel = torch.randn(batch_size, num_users, num_bs_ant, dtype=torch.complex64, device=device)
    channel = (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0, device=device))
    rzf = evaluate_baseline("rzf", channel, snr_db=snr_db, num_rf_chains=num_users)
    wmmse_iter_5 = evaluate_baseline("wmmse_iter_5", channel, snr_db=snr_db, num_rf_chains=num_users)
    noise_var = float(noise_variance_from_snr(snr_db).item())

    payload = {
        "sionna_import_ok": True,
        "sionna_version": info["sionna_version"],
        "bridge_status": "ok",
        "channel_shape": list(channel.shape),
        "snr_db": snr_db,
        "noise_var": noise_var,
        "rzf_mean_se": float(rzf["sum_rate"].mean().item()),
        "wmmse_iter_5_mean_se": float(wmmse_iter_5["sum_rate"].mean().item()),
        "notes": [
            "This bridge demo uses the existing PyTorch beamforming stack inside a Sionna-capable environment.",
            "It does not yet call a Sionna channel model and should not be interpreted as a Sionna end-to-end benchmark.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved bridge summary to {out_path}")


if __name__ == "__main__":
    main()
