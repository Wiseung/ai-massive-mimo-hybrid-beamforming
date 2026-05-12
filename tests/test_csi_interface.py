from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from beamforming.utils.csi_interface import ExtractedCSI


def _make_csi(h_f: torch.Tensor) -> ExtractedCSI:
    return ExtractedCSI(
        h_f=h_f,
        source="synthetic_project",
        source_component="unit_test",
        axes={"B": 0, "Nsc": 1, "K": 2, "Nt": 3},
        shape={"B": 2, "Nsc": 3, "K": 4, "Nt": 5},
        selected_ofdm_symbol=1,
        effective_subcarrier_indices=[1, 2, 3],
        num_users=4,
        num_bs_ant=5,
        num_subcarriers=3,
        project_h_f_assisted=True,
        extracted_h_f_used=False,
        full_native_only=False,
        metadata={"original_sionna_h_shape": None, "extraction_success": True, "fallback_reason": ""},
    )


def test_csi_interface_validate_shape_and_complex_dtype() -> None:
    h_f = torch.randn(2, 3, 4, 5, dtype=torch.float32) + 1j * torch.randn(2, 3, 4, 5, dtype=torch.float32)
    csi = _make_csi(h_f.to(torch.complex64))
    report = csi.validate(raise_on_error=False)
    assert report["valid"] is True
    assert report["is_complex"] is True
    assert report["all_finite"] is True


def test_csi_interface_summary_dict_fields() -> None:
    h_f = (torch.randn(2, 3, 4, 5) + 1j * torch.randn(2, 3, 4, 5)).to(torch.complex64)
    csi = _make_csi(h_f)
    summary = csi.summary_dict()
    assert summary["source"] == "synthetic_project"
    assert summary["axes"] == {"B": 0, "Nsc": 1, "K": 2, "Nt": 3}
    assert summary["h_f_shape"] == [2, 3, 4, 5]
    assert "validation" in summary


def test_csi_interface_no_nan_or_inf() -> None:
    h_f = torch.ones(2, 3, 4, 5, dtype=torch.complex64)
    csi = _make_csi(h_f)
    assert csi.validate(raise_on_error=False)["all_finite"] is True


def test_csi_interface_wrong_shape_raises_clear_error() -> None:
    h_f = (torch.randn(2, 3, 4) + 1j * torch.randn(2, 3, 4)).to(torch.complex64)
    csi = ExtractedCSI(
        h_f=h_f,
        source="synthetic_project",
        source_component="unit_test",
        axes={"B": 0, "Nsc": 1, "K": 2, "Nt": 3},
        shape={"B": 2, "Nsc": 3, "K": 4, "Nt": 5},
        selected_ofdm_symbol=1,
        effective_subcarrier_indices=[1, 2, 3],
        num_users=4,
        num_bs_ant=5,
        num_subcarriers=3,
        project_h_f_assisted=True,
        extracted_h_f_used=False,
        full_native_only=False,
        metadata={},
    )
    with pytest.raises(ValueError, match="expected_rank_4_h_f_got_3"):
        csi.validate()


def test_csi_interface_save_summary_json(tmp_path: Path) -> None:
    h_f = (torch.randn(2, 3, 4, 5) + 1j * torch.randn(2, 3, 4, 5)).to(torch.complex64)
    csi = _make_csi(h_f)
    out_path = tmp_path / "csi_summary.json"
    csi.save_summary_json(out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["h_f_shape"] == [2, 3, 4, 5]
