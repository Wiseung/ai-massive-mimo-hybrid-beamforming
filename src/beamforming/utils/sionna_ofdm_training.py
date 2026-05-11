"""Shared helpers for optional Sionna OFDM training and evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from beamforming.baselines.common import get_digital_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate
from beamforming.utils.sionna_phy_helpers import add_awgn_torch, try_import_sionna_ofdm, try_import_sionna_phy


def qpsk_map(bits: torch.Tensor) -> torch.Tensor:
    """Map bits with final dimension 2 to QPSK symbols."""
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    denom = torch.sqrt(torch.tensor(2.0, device=bits.device))
    return (real + 1j * imag) / denom


def hard_demod_qpsk(symbols: torch.Tensor) -> torch.Tensor:
    """Hard demodulate QPSK symbols to bits."""
    return torch.stack(
        [(symbols.real < 0).to(torch.int64), (symbols.imag < 0).to(torch.int64)],
        dim=-1,
    )


@dataclass
class OFDMContext:
    used_sionna_ofdm: bool
    used_sionna_channel: bool
    fallback_used: bool
    notes: list[str]
    tx_symbols: torch.Tensor
    bits: torch.Tensor
    num_data_symbols: int
    num_ofdm_symbols: int


def generate_qpsk_resource_grid(
    batch_size: int,
    num_subcarriers: int,
    num_users: int,
    device: torch.device,
    generator: torch.Generator,
) -> OFDMContext:
    """Create a QPSK OFDM grid using Sionna when possible, otherwise torch fallback."""
    bits = torch.randint(
        0,
        2,
        (batch_size, num_subcarriers, num_users, 2),
        generator=generator,
        dtype=torch.int64,
    ).to(device=device)
    fallback_symbols = qpsk_map(bits)
    notes: list[str] = []
    ofdm = try_import_sionna_ofdm()
    if ofdm["import_ok"]:
        try:
            rg = ofdm["ResourceGrid"](
                num_ofdm_symbols=1,
                fft_size=num_subcarriers,
                subcarrier_spacing=15_000.0,
                num_tx=num_users,
                num_streams_per_tx=1,
                num_guard_carriers=(0, 0),
                dc_null=False,
            )
            mapper = ofdm["ResourceGridMapper"](rg)
            bits_sionna = bits.permute(0, 2, 1, 3).unsqueeze(2)
            tx_symbols = qpsk_map(bits_sionna)
            tx_grid = mapper(tx_symbols)
            tx_grid = tx_grid.squeeze(2).squeeze(2).transpose(1, 2).contiguous().to(device=device)
            notes.append("Used real Sionna ResourceGrid/Mapper for OFDM symbol generation.")
            return OFDMContext(
                used_sionna_ofdm=True,
                used_sionna_channel=False,
                fallback_used=False,
                notes=notes,
                tx_symbols=tx_grid,
                bits=bits,
                num_data_symbols=int(rg.num_data_symbols),
                num_ofdm_symbols=1,
            )
        except Exception as exc:  # pragma: no cover - optional path
            notes.append(f"Sionna OFDM grid generation failed; used torch fallback: {type(exc).__name__}: {exc}")
    else:
        notes.append(f"Sionna OFDM components unavailable; used torch fallback: {ofdm['error']}")
    return OFDMContext(
        used_sionna_ofdm=False,
        used_sionna_channel=False,
        fallback_used=True,
        notes=notes,
        tx_symbols=fallback_symbols,
        bits=bits,
        num_data_symbols=num_subcarriers,
        num_ofdm_symbols=1,
    )


def simulate_multiuser_ofdm_link(
    channel_f: torch.Tensor,
    precoder: torch.Tensor,
    tx_symbols: torch.Tensor,
    noise_var: torch.Tensor,
    snr_db: torch.Tensor,
) -> dict[str, Any]:
    """Apply MU-MISO OFDM precoding, channel, and AWGN."""
    batch_size = channel_f.size(0)
    device = channel_f.device
    tx = tx_symbols.unsqueeze(-1)
    tx_signal = torch.matmul(precoder, tx).squeeze(-1)
    noiseless = torch.einsum("bsku,bsu->bsk", channel_f, tx_signal)

    phy = try_import_sionna_phy()
    used_sionna_channel = False
    fallback_used = False
    note: str | None = None
    if phy["import_ok"]:
        try:
            awgn = phy["AWGN"]()
            rx = awgn(noiseless, no=noise_var.reshape(batch_size, 1, 1).to(device=device, dtype=torch.float32))
            used_sionna_channel = True
        except Exception as exc:  # pragma: no cover - optional path
            rx = torch.empty_like(noiseless)
            for idx in range(batch_size):
                rx[idx], _ = add_awgn_torch(noiseless[idx], float(snr_db[idx].item()))
            fallback_used = True
            note = f"Sionna AWGN failed; used torch fallback: {type(exc).__name__}: {exc}"
    else:
        rx = torch.empty_like(noiseless)
        for idx in range(batch_size):
            rx[idx], _ = add_awgn_torch(noiseless[idx], float(snr_db[idx].item()))
        fallback_used = True
        note = f"Sionna AWGN unavailable; used torch fallback: {phy['error']}"
    return {
        "rx": rx,
        "tx_signal": tx_signal,
        "noiseless": noiseless,
        "used_sionna_channel": used_sionna_channel,
        "fallback_used": fallback_used,
        "note": note,
    }


def compute_link_metrics(
    channel_f: torch.Tensor,
    precoder: torch.Tensor,
    tx_symbols: torch.Tensor,
    rx_symbols: torch.Tensor,
    noise_var: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute differentiable OFDM link metrics."""
    if channel_f.ndim != 4 or precoder.ndim != 4:
        raise ValueError("channel_f and precoder must have shape (B, Nsc, K, Nt) and (B, Nsc, Nt, K).")

    batch_size, num_subcarriers, _, _ = channel_f.shape
    heff = torch.matmul(channel_f, precoder)
    signal = torch.abs(torch.diagonal(heff, dim1=-2, dim2=-1)) ** 2
    total = torch.abs(heff) ** 2
    interference = total.sum(dim=-1) - signal
    sinr = signal / (interference + noise_var.view(batch_size, 1, 1))
    sum_rate = torch.log2(1.0 + sinr).sum(dim=-1).mean(dim=-1)
    mse = torch.mean(torch.abs(rx_symbols - tx_symbols) ** 2, dim=(-2, -1))
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    power_violation = torch.abs(power - 1.0)

    baseline_rates = []
    for sc in range(num_subcarriers):
        baseline_rates.append(multi_user_downlink_sum_rate(channel_f[:, sc, :, :], precoder[:, sc, :, :], noise_var))
    rate_from_metric = torch.stack(baseline_rates, dim=1).mean(dim=1)

    return {
        "mean_sum_rate": rate_from_metric.mean(),
        "sum_rate_per_sample": rate_from_metric,
        "receive_mse": mse.mean(),
        "receive_mse_per_sample": mse,
        "power_norm": power.mean(),
        "power_violation": power_violation.mean(),
        "sinr_linear": sinr.mean(),
        "sinr_db": 10.0 * torch.log10(sinr.mean().clamp_min(1e-12)),
        "heff": heff,
        "signal_power": signal.mean(),
        "interference_power": interference.mean(),
        "sum_rate_from_sinr": sum_rate.mean(),
    }


def build_baseline_precoder_stack(method: str, channel_f: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
    """Build a per-subcarrier baseline precoder stack."""
    precoders = []
    for sc in range(channel_f.size(1)):
        precoders.append(get_digital_precoder(method, channel_f[:, sc, :, :], noise_var=noise_var))
    return torch.stack(precoders, dim=1)


def run_model_forward(model: torch.nn.Module, channel_f: torch.Tensor, snr_db: torch.Tensor) -> dict[str, Any]:
    """Normalize model outputs into a shared dict with a precoder field."""
    outputs = model(channel_f, snr_db=snr_db)
    if isinstance(outputs, dict):
        if "precoder" not in outputs:
            raise ValueError("Structured OFDM model output must include a 'precoder' field.")
        return outputs
    if torch.is_tensor(outputs):
        return {"precoder": outputs}
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")
