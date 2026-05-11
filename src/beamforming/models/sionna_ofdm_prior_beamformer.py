"""Communication-prior beamformers for optional Sionna OFDM training."""

from __future__ import annotations

import torch

from beamforming.baselines.common import get_digital_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr
from beamforming.models.constraints import power_normalization


def _zero_init_linear(linear: torch.nn.Linear) -> None:
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)


def _flatten_complex(x: torch.Tensor) -> torch.Tensor:
    batch = x.size(0)
    return torch.view_as_real(x).reshape(batch, -1)


def _ofdm_rate_trace(channel_f: torch.Tensor, precoder_f: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
    rates = []
    for sc in range(channel_f.size(1)):
        rates.append(multi_user_downlink_sum_rate(channel_f[:, sc, :, :], precoder_f[:, sc, :, :], noise_var))
    return torch.stack(rates, dim=1).mean(dim=1)


class _SubcarrierResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        condition_on_snr: bool,
    ) -> None:
        super().__init__()
        self.condition_on_snr = condition_on_snr
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim + (1 if condition_on_snr else 0), hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        final = self.net[-1]
        if isinstance(final, torch.nn.Linear):
            _zero_init_linear(final)

    def forward(self, features: torch.Tensor, snr_db: torch.Tensor | None = None) -> torch.Tensor:
        if self.condition_on_snr:
            if snr_db is None:
                raise ValueError("snr_db must be provided when condition_on_snr=True.")
            if snr_db.ndim == 0:
                snr_db = snr_db.unsqueeze(0)
            features = torch.cat([features, snr_db.reshape(features.size(0), 1) / 20.0], dim=-1)
        return self.net(features)


class SionnaOFDMResidualRZFBeamformer(torch.nn.Module):
    """Residual RZF refiner for OFDM multi-subcarrier training."""

    def __init__(
        self,
        num_users: int,
        num_bs_ant: int,
        hidden_dim: int = 128,
        alpha_init: float = 0.1,
        learnable_alpha: bool = True,
        condition_on_snr: bool = True,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.condition_on_snr = condition_on_snr
        self.learnable_alpha = learnable_alpha
        self.model_name = "sionna_ofdm_residual_rzf"
        feature_dim = 4 * num_users * num_bs_ant
        output_dim = 2 * num_bs_ant * num_users
        self.delta_head = _SubcarrierResidualBlock(
            in_dim=feature_dim,
            out_dim=output_dim,
            hidden_dim=hidden_dim,
            condition_on_snr=condition_on_snr,
        )
        if learnable_alpha:
            init = torch.tensor(float(alpha_init), dtype=torch.float32)
            self.log_alpha = torch.nn.Parameter(torch.log(torch.expm1(init) + 1e-8))
        else:
            self.register_buffer("fixed_alpha", torch.tensor(float(alpha_init), dtype=torch.float32))

    def _alpha(self) -> torch.Tensor:
        if self.learnable_alpha:
            return torch.nn.functional.softplus(self.log_alpha)
        return self.fixed_alpha

    def _base_precoder(self, channel_f: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
        precoders = []
        for sc in range(channel_f.size(1)):
            precoders.append(get_digital_precoder("rzf", channel_f[:, sc, :, :], noise_var=noise_var))
        return torch.stack(precoders, dim=1)

    def forward(self, channel_f: torch.Tensor, snr_db: torch.Tensor | None = None) -> dict[str, torch.Tensor | str]:
        if snr_db is None:
            raise ValueError("snr_db must be provided for SionnaOFDMResidualRZFBeamformer.")
        if channel_f.ndim != 4:
            raise ValueError("channel_f must have shape (B, Nsc, K, Nt).")
        batch, num_sc, _, _ = channel_f.shape
        noise_var = noise_variance_from_snr(snr_db).to(channel_f.device)
        base_precoder = self._base_precoder(channel_f, noise_var)

        flat_channel = channel_f.reshape(batch * num_sc, self.num_users, self.num_bs_ant)
        flat_base = base_precoder.reshape(batch * num_sc, self.num_bs_ant, self.num_users)
        features = torch.cat([_flatten_complex(flat_channel), _flatten_complex(flat_base)], dim=-1)
        snr_feature = snr_db.repeat_interleave(num_sc).to(device=channel_f.device, dtype=torch.float32)
        delta_raw = self.delta_head(features, snr_feature if self.condition_on_snr else None)
        delta_real = delta_raw.reshape(batch, num_sc, self.num_bs_ant, self.num_users, 2)
        delta_precoder = torch.complex(delta_real[..., 0], delta_real[..., 1])
        alpha = self._alpha().to(channel_f.device)
        precoder = power_normalization(base_precoder + alpha * delta_precoder)
        return {
            "precoder": precoder,
            "base_precoder": base_precoder,
            "delta_precoder": delta_precoder,
            "alpha": alpha,
            "model_name": self.model_name,
            "base_method": "rzf",
        }


class SionnaOFDMUnfoldedLiteBeamformer(torch.nn.Module):
    """Structured OFDM refiner with RZF or few-iteration WMMSE initialization."""

    def __init__(
        self,
        num_users: int,
        num_bs_ant: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        init_method: str = "wmmse_iter_2",
        learnable_step_size: bool = True,
        step_size_init: float = 0.05,
        condition_on_snr: bool = True,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_layers = num_layers
        self.init_method = init_method
        self.learnable_step_size = learnable_step_size
        self.condition_on_snr = condition_on_snr
        self.model_name = "sionna_ofdm_unfolded_lite"
        feature_dim = 4 * num_users * num_bs_ant
        output_dim = 2 * num_bs_ant * num_users
        self.layers = torch.nn.ModuleList(
            [
                _SubcarrierResidualBlock(
                    in_dim=feature_dim,
                    out_dim=output_dim,
                    hidden_dim=hidden_dim,
                    condition_on_snr=condition_on_snr,
                )
                for _ in range(num_layers)
            ]
        )
        init = torch.full((num_layers,), float(step_size_init), dtype=torch.float32)
        if learnable_step_size:
            self.log_steps = torch.nn.Parameter(torch.log(torch.expm1(init) + 1e-8))
        else:
            self.register_buffer("fixed_steps", init)

    def _steps(self) -> torch.Tensor:
        if self.learnable_step_size:
            return torch.nn.functional.softplus(self.log_steps)
        return self.fixed_steps

    def _base_precoder(self, channel_f: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
        precoders = []
        for sc in range(channel_f.size(1)):
            precoders.append(get_digital_precoder(self.init_method, channel_f[:, sc, :, :], noise_var=noise_var))
        return torch.stack(precoders, dim=1)

    def forward(self, channel_f: torch.Tensor, snr_db: torch.Tensor | None = None) -> dict[str, torch.Tensor | str | int]:
        if snr_db is None:
            raise ValueError("snr_db must be provided for SionnaOFDMUnfoldedLiteBeamformer.")
        if channel_f.ndim != 4:
            raise ValueError("channel_f must have shape (B, Nsc, K, Nt).")
        batch, num_sc, _, _ = channel_f.shape
        noise_var = noise_variance_from_snr(snr_db).to(channel_f.device)
        base_precoder = self._base_precoder(channel_f, noise_var)
        current = base_precoder
        layer_rates: list[torch.Tensor] = []
        snr_feature = snr_db.repeat_interleave(num_sc).to(device=channel_f.device, dtype=torch.float32)

        for layer, step_size in zip(self.layers, self._steps()):
            flat_channel = channel_f.reshape(batch * num_sc, self.num_users, self.num_bs_ant)
            flat_current = current.reshape(batch * num_sc, self.num_bs_ant, self.num_users)
            features = torch.cat([_flatten_complex(flat_channel), _flatten_complex(flat_current)], dim=-1)
            delta_raw = layer(features, snr_feature if self.condition_on_snr else None)
            delta_real = delta_raw.reshape(batch, num_sc, self.num_bs_ant, self.num_users, 2)
            delta_precoder = torch.complex(delta_real[..., 0], delta_real[..., 1])
            current = power_normalization(current + step_size.to(channel_f.device) * delta_precoder)
            layer_rates.append(_ofdm_rate_trace(channel_f, current, noise_var))

        return {
            "precoder": current,
            "base_precoder": base_precoder,
            "delta_precoder": current - base_precoder,
            "layer_sum_rates": torch.stack(layer_rates, dim=1) if layer_rates else torch.empty(batch, 0, device=channel_f.device),
            "step_sizes": self._steps().to(channel_f.device),
            "model_name": self.model_name,
            "init_method": self.init_method,
            "num_layers": self.num_layers,
        }
