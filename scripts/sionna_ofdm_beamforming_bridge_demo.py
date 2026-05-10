#!/usr/bin/env python
"""OFDM-like beamforming bridge demo using Sionna OFDM grid components when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.baselines.common import get_digital_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_phy_helpers import add_awgn_torch, try_import_sionna_ofdm, try_import_sionna_phy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _rand_qpsk(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    bits = torch.randint(0, 2, shape + (2,), device=device)
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    return (real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=device))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_info = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_subcarriers = 8
    num_users = 4
    num_bs_ant = 16
    snr_db = 10.0
    noise_var = float(noise_variance_from_snr(snr_db).item())

    ofdm = try_import_sionna_ofdm()
    phy = try_import_sionna_phy()
    used_sionna_ofdm = False
    used_sionna_channel = False
    fallback_used = False
    notes: list[str] = []

    tx_symbols = _rand_qpsk((batch_size, num_subcarriers, num_users), device)
    channel_f = torch.randn(batch_size, num_subcarriers, num_users, num_bs_ant, dtype=torch.complex64, device=device)
    channel_f = (channel_f + 1j * torch.randn_like(channel_f)) / torch.sqrt(torch.tensor(2.0, device=device))

    rzf_precoders = []
    wmmse_precoders = []
    for sc in range(num_subcarriers):
        h_sc = channel_f[:, sc, :, :]
        rzf_precoders.append(get_digital_precoder("rzf", h_sc, noise_var=noise_var))
        wmmse_precoders.append(get_digital_precoder("wmmse_iter_5", h_sc, noise_var=noise_var))
    rzf_precoder = torch.stack(rzf_precoders, dim=1)
    wmmse_precoder = torch.stack(wmmse_precoders, dim=1)

    if ofdm["import_ok"]:
        try:
            rg = ofdm["ResourceGrid"](
                num_ofdm_symbols=1,
                fft_size=num_subcarriers,
                subcarrier_spacing=15_000.0,
                num_tx=1,
                num_streams_per_tx=1,
                num_guard_carriers=(0, 0),
                dc_null=False,
            )
            mapper = ofdm["ResourceGridMapper"](rg)
            sm = ofdm["StreamManagement"](np.array([[1]]), num_streams_per_tx=1)
            demapper = ofdm["ResourceGridDemapper"](rg, sm)
            _ = demapper  # only to prove constructibility
            flat_qpsk = tx_symbols[:, :, 0].reshape(batch_size, 1, 1, rg.num_data_symbols)
            _ = mapper(flat_qpsk)
            used_sionna_ofdm = True
            notes.append("Used real Sionna OFDM ResourceGrid/Mapper in the OFDM bridge setup.")
        except Exception as exc:  # pragma: no cover
            fallback_used = True
            notes.append(f"Sionna OFDM grid path failed; continuing with torch grid fallback: {type(exc).__name__}: {exc}")
    else:
        fallback_used = True
        notes.append(f"Sionna OFDM components unavailable; using torch grid fallback: {ofdm['error']}")

    def run_link(precoder_stack: torch.Tensor) -> tuple[float, float, list[float]]:
        nonlocal used_sionna_channel, fallback_used, notes
        tx = tx_symbols.unsqueeze(-1)
        tx_signal = torch.matmul(precoder_stack, tx).squeeze(-1)
        noiseless = torch.einsum("bsku,bsu->bsk", channel_f, tx_signal)
        if phy["import_ok"]:
            try:
                awgn = phy["AWGN"]()
                rx = awgn(noiseless, no=torch.full((batch_size, 1, 1), noise_var, dtype=torch.float32, device=device))
                used_sionna_channel = True
            except Exception as exc:  # pragma: no cover
                rx, _ = add_awgn_torch(noiseless, snr_db)
                fallback_used = True
                notes.append(f"Sionna AWGN failed in OFDM bridge; used torch fallback: {type(exc).__name__}: {exc}")
        else:
            rx, _ = add_awgn_torch(noiseless, snr_db)
            fallback_used = True
            notes.append(f"Sionna AWGN unavailable in OFDM bridge; used torch fallback: {phy['error']}")
        mse = float(torch.mean(torch.abs(rx - noiseless) ** 2).item())
        per_sc_rates = []
        for sc in range(num_subcarriers):
            rate = multi_user_downlink_sum_rate(channel_f[:, sc, :, :], precoder_stack[:, sc, :, :], noise_var)
            per_sc_rates.append(float(rate.mean().item()))
        avg_rate = float(sum(per_sc_rates) / len(per_sc_rates))
        return mse, avg_rate, per_sc_rates

    rzf_mse, rzf_rate, rzf_sc = run_link(rzf_precoder)
    wmmse_mse, wmmse_rate, wmmse_sc = run_link(wmmse_precoder)

    summary = {
        "sionna_import_ok": env_info["sionna_import_ok"],
        "sionna_version": env_info["sionna_version"],
        "used_sionna_ofdm": used_sionna_ofdm,
        "used_sionna_channel": used_sionna_channel,
        "fallback_used": fallback_used,
        "device": str(device),
        "channel_shape": list(channel_f.shape),
        "num_subcarriers": num_subcarriers,
        "snr_db": snr_db,
        "rzf_ofdm_average_mse": rzf_mse,
        "wmmse_iter_5_ofdm_average_mse": wmmse_mse,
        "rzf_ofdm_average_sum_rate": rzf_rate,
        "wmmse_iter_5_ofdm_average_sum_rate": wmmse_rate,
        "rzf_per_subcarrier_se_mean": float(np.mean(rzf_sc)),
        "rzf_per_subcarrier_se_std": float(np.std(rzf_sc)),
        "wmmse_iter_5_per_subcarrier_se_mean": float(np.mean(wmmse_sc)),
        "wmmse_iter_5_per_subcarrier_se_std": float(np.std(wmmse_sc)),
        "demo_status": "ok",
        "notes": notes
        + [
            "This is an OFDM-like bridge demo only.",
            "It is not a full Sionna end-to-end link, not Sionna RT, and not a 5G NR full-stack system.",
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Sionna OFDM beamforming bridge summary to {out_path}")


if __name__ == "__main__":
    main()
