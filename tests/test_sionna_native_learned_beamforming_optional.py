from __future__ import annotations

from pathlib import Path

import pytest
import torch

from beamforming.utils.csi_interface import ExtractedCSI
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_learned_beamforming import (
    default_checkpoint_path,
    infer_learned_precoder,
    load_learned_beamformer_checkpoint,
)


def _make_csi(h_f: torch.Tensor) -> ExtractedCSI:
    return ExtractedCSI(
        h_f=h_f,
        source="synthetic_project",
        source_component="unit_test",
        axes={"B": 0, "Nsc": 1, "K": 2, "Nt": 3},
        shape={"B": int(h_f.size(0)), "Nsc": int(h_f.size(1)), "K": int(h_f.size(2)), "Nt": int(h_f.size(3))},
        selected_ofdm_symbol=1,
        effective_subcarrier_indices=list(range(int(h_f.size(1)))),
        num_users=int(h_f.size(2)),
        num_bs_ant=int(h_f.size(3)),
        num_subcarriers=int(h_f.size(1)),
        project_h_f_assisted=True,
        extracted_h_f_used=False,
        full_native_only=False,
        metadata={"original_sionna_h_shape": None, "extraction_success": True, "fallback_reason": ""},
    )


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_native_learned_beamforming_adapter_shape_and_teacher_flag() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = default_checkpoint_path("learned_residual_rzf", repo_root)
    if not ckpt.exists():
        pytest.skip("Residual-RZF checkpoint not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_learned_beamformer_checkpoint(ckpt, device, method_name="learned_residual_rzf")
    h_f = torch.randn(2, 8, 4, 16, dtype=torch.complex64, device=device)
    h_f = (h_f + 1j * torch.randn_like(h_f)) / torch.sqrt(torch.tensor(2.0, device=device))
    snr_db = torch.tensor([0.0, 10.0], dtype=torch.float32, device=device)
    precoder, meta, _ = infer_learned_precoder(bundle, h_f, snr_db, native_receiver_path=True)
    assert precoder.shape == (2, 8, 16, 4)
    assert torch.isfinite(precoder.real).all()
    assert torch.isfinite(precoder.imag).all()
    assert meta["teacher_used_during_inference"] is False
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_native_learned_wmmse_distill_teacher_flag_false() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = default_checkpoint_path("learned_residual_wmmse_distill", repo_root)
    if not ckpt.exists():
        pytest.skip("Residual-WMMSE-distill checkpoint not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_learned_beamformer_checkpoint(ckpt, device, method_name="learned_residual_wmmse_distill")
    h_f = torch.randn(1, 8, 4, 16, dtype=torch.complex64, device=device)
    h_f = (h_f + 1j * torch.randn_like(h_f)) / torch.sqrt(torch.tensor(2.0, device=device))
    csi = _make_csi(h_f)
    snr_db = torch.tensor([10.0], dtype=torch.float32, device=device)
    _, meta, _ = infer_learned_precoder(bundle, csi, snr_db, native_receiver_path=True)
    assert meta["teacher_used_during_inference"] is False
    assert meta["uses_project_h_f_input"] is True
    assert meta["native_receiver_path"] is True
    assert meta["csi_interface_used"] is True
    assert meta["input_type"] == "ExtractedCSI"
