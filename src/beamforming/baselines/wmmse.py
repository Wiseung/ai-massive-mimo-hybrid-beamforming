"""Small-scale MU-MISO WMMSE baseline."""

from __future__ import annotations

import warnings

import torch

from beamforming.baselines.mrt import mrt_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate
from beamforming.models.constraints import power_normalization


def _power_project(precoder: torch.Tensor, total_power: float) -> torch.Tensor:
    return power_normalization(precoder, total_power=total_power)


def _batch_eye(size: int, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(size, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)


def _solve_with_mu(
    system_base: torch.Tensor,
    rhs: torch.Tensor,
    total_power: float,
    reg: float,
    max_bisection: int = 20,
) -> torch.Tensor:
    batch_size, size, _ = system_base.shape
    device = system_base.device
    dtype = system_base.dtype
    eye = _batch_eye(size, batch_size, dtype=dtype, device=device)

    def solve_for(mu: torch.Tensor) -> torch.Tensor:
        mu_term = mu.view(-1, 1, 1).to(device=device, dtype=torch.float32)
        return torch.linalg.solve(system_base + (mu_term + reg) * eye, rhs)

    candidate0 = solve_for(torch.zeros(batch_size, device=device))
    power0 = (torch.abs(candidate0) ** 2).sum(dim=(-2, -1))
    if torch.all(power0 <= total_power * (1.0 + 1e-4)):
        return _power_project(candidate0, total_power)

    low = torch.zeros(batch_size, device=device)
    high = torch.ones(batch_size, device=device)
    active = power0 > total_power

    for _ in range(12):
        if not torch.any(active):
            break
        candidate_high = solve_for(high)
        power_high = (torch.abs(candidate_high) ** 2).sum(dim=(-2, -1))
        grow = active & (power_high > total_power)
        if not torch.any(grow):
            break
        high = torch.where(grow, high * 2.0, high)

    for _ in range(max_bisection):
        mid = 0.5 * (low + high)
        candidate_mid = solve_for(mid)
        power_mid = (torch.abs(candidate_mid) ** 2).sum(dim=(-2, -1))
        too_large = power_mid > total_power
        low = torch.where(too_large, mid, low)
        high = torch.where(too_large, high, mid)

    solved = solve_for(high)
    return _power_project(solved, total_power)


def wmmse_precoder(
    channel: torch.Tensor,
    noise_var: float | torch.Tensor,
    total_power: float = 1.0,
    max_iter: int = 30,
    tol: float = 1e-5,
    reg: float = 1e-8,
) -> torch.Tensor:
    """Run a numerically-stable MU-MISO WMMSE iteration.

    Args:
        channel: complex channel with shape ``(B, K, Nt)``.
        noise_var: scalar or per-sample noise variance.
        total_power: transmit power constraint.
        max_iter: maximum alternating-optimization iterations.
        tol: stop when the normalized precoder change falls below this value.
        reg: diagonal regularization used in the linear solve.
    """
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    if channel.ndim != 3:
        raise ValueError("channel must have shape (B, K, Nt)")

    batch_size, num_users, num_bs_ant = channel.shape
    device = channel.device
    dtype = channel.dtype
    noise = torch.as_tensor(noise_var, dtype=torch.float32, device=device).view(-1)
    if noise.numel() == 1:
        noise = noise.repeat(batch_size)
    if noise.numel() != batch_size:
        raise ValueError("noise_var must be scalar or have batch size entries")

    precoder = mrt_precoder(channel, total_power=total_power)
    best_precoder = precoder.clone()
    best_rate = multi_user_downlink_sum_rate(channel, precoder, noise).detach()

    for iter_idx in range(max_iter):
        prev = precoder
        heff = torch.matmul(channel, precoder)
        diag = torch.diagonal(heff, dim1=-2, dim2=-1)
        total_power_rx = (torch.abs(heff) ** 2).sum(dim=-1) + noise.unsqueeze(-1)
        signal_power = torch.abs(diag) ** 2
        mse = (1.0 - signal_power / total_power_rx.clamp_min(1e-8)).clamp_min(1e-8)
        receiver = diag.conj() / total_power_rx.clamp_min(1e-8)
        weights = (1.0 / mse).clamp(max=1e8)

        hu = channel.transpose(-2, -1).conj()
        weighted_h = hu * (weights * torch.abs(receiver) ** 2).unsqueeze(1)
        system = torch.matmul(weighted_h, channel)
        target = hu * (weights * receiver.conj()).unsqueeze(1)

        try:
            precoder = _solve_with_mu(system, target, total_power=total_power, reg=reg)
        except RuntimeError as exc:
            warnings.warn(
                f"WMMSE linear solve failed at iteration {iter_idx}: {exc}. Falling back to previous iterate.",
                stacklevel=2,
            )
            precoder = prev

        if torch.isnan(precoder).any() or torch.isinf(precoder).any():
            warnings.warn(
                f"WMMSE produced invalid values at iteration {iter_idx}; falling back to previous iterate.",
                stacklevel=2,
            )
            precoder = prev
            break

        current_rate = multi_user_downlink_sum_rate(channel, precoder, noise).detach()
        improve_mask = current_rate > best_rate
        best_rate = torch.where(improve_mask, current_rate, best_rate)
        best_precoder = torch.where(improve_mask.view(-1, 1, 1), precoder, best_precoder)
        delta = torch.linalg.norm((precoder - prev).reshape(batch_size, -1), dim=-1)
        base = torch.linalg.norm(prev.reshape(batch_size, -1), dim=-1).clamp_min(1e-8)
        if torch.all(delta / base < tol):
            break

    return _power_project(best_precoder, total_power=total_power)
