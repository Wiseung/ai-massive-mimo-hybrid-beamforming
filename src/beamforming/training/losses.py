"""Training losses for beamforming models."""

from __future__ import annotations

import torch

from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr


def beamforming_loss(
    channel: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    snr_db: torch.Tensor,
    lambda_power: float = 1e-2,
    lambda_const: float = 1e-2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Negative sum-rate with soft constraint penalties."""
    precoder = outputs["precoder"]
    noise_var = noise_variance_from_snr(snr_db).to(channel.device)
    if noise_var.ndim == 0:
        noise_var = noise_var.repeat(channel.size(0))
    sum_rate = multi_user_downlink_sum_rate(channel, precoder, noise_var)
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    power_violation = torch.mean((power - 1.0) ** 2)

    const_violation = torch.tensor(0.0, device=channel.device)
    if "analog_precoder" in outputs:
        analog = outputs["analog_precoder"]
        target = torch.full_like(torch.abs(analog), fill_value=1.0 / (analog.size(-2) ** 0.5))
        const_violation = torch.mean((torch.abs(analog) - target) ** 2)

    loss = -sum_rate.mean() + lambda_power * power_violation + lambda_const * const_violation
    stats = {
        "loss": loss.detach(),
        "sum_rate": sum_rate.mean().detach(),
        "power_violation": power_violation.detach(),
        "constant_modulus_violation": const_violation.detach(),
    }
    return loss, stats
