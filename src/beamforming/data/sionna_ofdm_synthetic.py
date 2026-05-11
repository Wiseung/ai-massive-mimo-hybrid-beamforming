"""Synthetic OFDM channel generation for optional Sionna training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from beamforming.metrics.sum_rate import noise_variance_from_snr


def _randn_complex(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    real = torch.randn(shape, generator=generator, dtype=torch.float32)
    imag = torch.randn(shape, generator=generator, dtype=torch.float32)
    return torch.complex(real, imag)


def qpsk_symbols(
    batch_size: int,
    num_subcarriers: int,
    num_users: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate QPSK symbols and the underlying bits."""
    bits = torch.randint(
        0,
        2,
        (batch_size, num_subcarriers, num_users, 2),
        generator=generator,
        dtype=torch.int64,
    )
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    symbols = (real + 1j * imag) / torch.sqrt(torch.tensor(2.0))
    return symbols.to(device=device), bits.to(device=device)


@dataclass
class SionnaOFDMSyntheticConfig:
    batch_size: int
    num_subcarriers: int
    num_users: int
    num_bs_ant: int
    snr_db_choices: Sequence[float]
    channel_model: str = "rayleigh"
    sparse_mmwave_like: bool = False
    num_paths: int = 3
    seed: int = 42


class SionnaOFDMSyntheticGenerator:
    """Batch generator for synthetic OFDM channel training."""

    def __init__(self, config: SionnaOFDMSyntheticConfig) -> None:
        self.config = config
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(config.seed))
        if not config.snr_db_choices:
            raise ValueError("snr_db_choices must not be empty.")

    def _sample_snr(self, batch_size: int) -> torch.Tensor:
        choices = torch.tensor(list(self.config.snr_db_choices), dtype=torch.float32)
        indices = torch.randint(0, choices.numel(), (batch_size,), generator=self.generator)
        return choices[indices]

    def _rayleigh_channel(self, batch_size: int) -> torch.Tensor:
        shape = (batch_size, self.config.num_subcarriers, self.config.num_users, self.config.num_bs_ant)
        channel = _randn_complex(shape, self.generator)
        return channel / torch.sqrt(torch.tensor(2.0))

    def _sparse_mmwave_channel(self, batch_size: int) -> torch.Tensor:
        num_sc = self.config.num_subcarriers
        num_users = self.config.num_users
        num_ant = self.config.num_bs_ant
        num_paths = max(1, int(self.config.num_paths))
        antenna_idx = torch.arange(num_ant, dtype=torch.float32).view(1, 1, 1, num_ant)
        subcarrier_idx = torch.arange(num_sc, dtype=torch.float32).view(1, num_sc, 1, 1, 1)
        angles = torch.rand((batch_size, num_users, num_paths), generator=self.generator) * torch.pi - (torch.pi / 2.0)
        delays = torch.rand((batch_size, num_users, num_paths), generator=self.generator)
        path_gain = _randn_complex((batch_size, num_users, num_paths), self.generator) / torch.sqrt(torch.tensor(2.0 * num_paths))

        steering = torch.exp(1j * torch.pi * antenna_idx * torch.sin(angles).unsqueeze(-1))
        freq_phase = torch.exp(-1j * 2.0 * torch.pi * subcarrier_idx * delays.unsqueeze(1).unsqueeze(-1) / max(num_sc, 1))
        channel = (freq_phase * path_gain.unsqueeze(1).unsqueeze(-1) * steering.unsqueeze(1)).sum(dim=3)
        return channel.to(torch.complex64)

    def sample_batch(
        self,
        batch_size: int | None = None,
        device: str | torch.device = "cpu",
        return_symbols: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Sample one synthetic OFDM batch."""
        actual_batch = int(batch_size or self.config.batch_size)
        device_obj = torch.device(device)
        use_sparse = self.config.sparse_mmwave_like or self.config.channel_model.lower() in {"sparse", "mmwave", "mmwave_like"}
        if use_sparse:
            channel_f = self._sparse_mmwave_channel(actual_batch)
        else:
            channel_f = self._rayleigh_channel(actual_batch)
        snr_db = self._sample_snr(actual_batch)
        noise_var = noise_variance_from_snr(snr_db)
        payload = {
            "H_f": channel_f.to(device=device_obj, dtype=torch.complex64),
            "snr_db": snr_db.to(device=device_obj, dtype=torch.float32),
            "noise_var": noise_var.to(device=device_obj, dtype=torch.float32),
        }
        if return_symbols:
            symbols, bits = qpsk_symbols(
                batch_size=actual_batch,
                num_subcarriers=self.config.num_subcarriers,
                num_users=self.config.num_users,
                generator=self.generator,
                device=device_obj,
            )
            payload["symbols"] = symbols
            payload["bits"] = bits
        return payload
