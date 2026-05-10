#!/usr/bin/env python
"""Small beamforming link demo that optionally uses Sionna PHY AWGN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.baselines.common import get_digital_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_phy_helpers import add_awgn_torch, try_import_sionna_phy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_info = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_users = 4
    num_bs_ant = 16
    snr_db = 10.0
    noise_var = float(noise_variance_from_snr(snr_db).item())

    channel = torch.randn(batch_size, num_users, num_bs_ant, dtype=torch.complex64, device=device)
    channel = (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0, device=device))
    tx_symbols = (
        torch.randn(batch_size, num_users, 1, dtype=torch.complex64, device=device)
        + 1j * torch.randn(batch_size, num_users, 1, dtype=torch.complex64, device=device)
    ) / torch.sqrt(torch.tensor(2.0, device=device))

    rzf = get_digital_precoder("rzf", channel, noise_var=noise_var)
    wmmse_iter_5 = get_digital_precoder("wmmse_iter_5", channel, noise_var=noise_var)

    phy = try_import_sionna_phy()
    used_sionna_phy = False
    fallback_used = False
    notes: list[str] = []

    def transmit_and_measure(precoder: torch.Tensor) -> tuple[torch.Tensor, float]:
        nonlocal used_sionna_phy, fallback_used, notes
        tx_signal = torch.matmul(precoder, tx_symbols)
        noiseless_rx = torch.matmul(channel, tx_signal).squeeze(-1)
        if phy["import_ok"]:
            try:
                awgn = phy["AWGN"]()
                no = torch.full((batch_size, 1), noise_var, dtype=torch.float32, device=device)
                rx = awgn(noiseless_rx, no=no)
                used_sionna_phy = True
            except Exception as exc:  # pragma: no cover
                rx, _ = add_awgn_torch(noiseless_rx, snr_db)
                fallback_used = True
                notes.append(f"Sionna PHY AWGN failed in link demo; used torch fallback: {type(exc).__name__}: {exc}")
        else:
            rx, _ = add_awgn_torch(noiseless_rx, snr_db)
            fallback_used = True
            notes.append(f"Sionna PHY AWGN unavailable in link demo; used torch fallback: {phy['error']}")
        mse = float(torch.mean(torch.abs(rx - noiseless_rx) ** 2).item())
        return rx, mse

    _, rzf_rx_mse = transmit_and_measure(rzf)
    _, wmmse_rx_mse = transmit_and_measure(wmmse_iter_5)
    rzf_rate = float(multi_user_downlink_sum_rate(channel, rzf, noise_var).mean().item())
    wmmse_rate = float(multi_user_downlink_sum_rate(channel, wmmse_iter_5, noise_var).mean().item())

    summary = {
        "sionna_import_ok": env_info["sionna_import_ok"],
        "sionna_version": env_info["sionna_version"],
        "used_sionna_phy": used_sionna_phy,
        "fallback_used": fallback_used,
        "device": str(device),
        "channel_shape": list(channel.shape),
        "snr_db": snr_db,
        "noise_var": noise_var,
        "rzf_receive_mse": rzf_rx_mse,
        "wmmse_iter_5_receive_mse": wmmse_rx_mse,
        "rzf_sum_rate": rzf_rate,
        "wmmse_iter_5_sum_rate": wmmse_rate,
        "approx_effective_sinr_db_rzf": 10.0 * torch.log10(torch.tensor(1.0 / max(rzf_rx_mse, 1e-12))).item(),
        "approx_effective_sinr_db_wmmse_iter_5": 10.0 * torch.log10(torch.tensor(1.0 / max(wmmse_rx_mse, 1e-12))).item(),
        "demo_status": "ok",
        "notes": notes
        + [
            "This is a Sionna-compatible beamforming link smoke demo.",
            "It is not a full Sionna end-to-end pipeline, not Sionna RT, and not a 5G NR full-stack demo.",
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Sionna PHY beamforming link summary to {out_path}")


if __name__ == "__main__":
    main()
