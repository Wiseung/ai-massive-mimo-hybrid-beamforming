from __future__ import annotations

import torch

from beamforming.data.sionna_ofdm_synthetic import SionnaOFDMSyntheticConfig, SionnaOFDMSyntheticGenerator


def _make_generator(seed: int = 7) -> SionnaOFDMSyntheticGenerator:
    return SionnaOFDMSyntheticGenerator(
        SionnaOFDMSyntheticConfig(
            batch_size=4,
            num_subcarriers=8,
            num_users=4,
            num_bs_ant=16,
            snr_db_choices=[0.0, 5.0, 10.0],
            seed=seed,
        )
    )


def test_sionna_ofdm_synthetic_shapes_and_noise() -> None:
    generator = _make_generator()
    batch = generator.sample_batch(device="cpu", return_symbols=True)
    assert batch["H_f"].shape == (4, 8, 4, 16)
    assert batch["snr_db"].shape == (4,)
    assert batch["noise_var"].shape == (4,)
    assert batch["symbols"].shape == (4, 8, 4)
    assert batch["bits"].shape == (4, 8, 4, 2)
    assert torch.isfinite(batch["H_f"].real).all()
    assert torch.isfinite(batch["H_f"].imag).all()
    assert torch.isfinite(batch["noise_var"]).all()
    assert (batch["noise_var"] > 0).all()


def test_sionna_ofdm_synthetic_reproducibility() -> None:
    batch_a = _make_generator(seed=123).sample_batch(device="cpu", return_symbols=True)
    batch_b = _make_generator(seed=123).sample_batch(device="cpu", return_symbols=True)
    assert torch.allclose(batch_a["H_f"], batch_b["H_f"])
    assert torch.equal(batch_a["bits"], batch_b["bits"])
    assert torch.allclose(batch_a["snr_db"], batch_b["snr_db"])
    assert torch.allclose(batch_a["noise_var"], batch_b["noise_var"])


def test_sionna_ofdm_sparse_mmwave_like_channel_has_valid_values() -> None:
    generator = SionnaOFDMSyntheticGenerator(
        SionnaOFDMSyntheticConfig(
            batch_size=2,
            num_subcarriers=8,
            num_users=4,
            num_bs_ant=16,
            snr_db_choices=[10.0],
            seed=5,
            sparse_mmwave_like=True,
            num_paths=2,
        )
    )
    batch = generator.sample_batch(device="cpu", return_symbols=False)
    assert batch["H_f"].shape == (2, 8, 4, 16)
    assert torch.isfinite(batch["H_f"].real).all()
    assert torch.isfinite(batch["H_f"].imag).all()
