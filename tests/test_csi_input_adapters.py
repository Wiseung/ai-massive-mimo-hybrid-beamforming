from __future__ import annotations

import pytest
import torch

from beamforming.utils.csi_interface import (
    ExtractedCSI,
    as_project_h_f,
    compare_csi_inputs,
    require_extracted_csi,
    summarize_csi_input,
)


def _make_h_f() -> torch.Tensor:
    real = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    imag = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    return (real + 1j * imag).to(torch.complex64)


def _make_csi(h_f: torch.Tensor) -> ExtractedCSI:
    return ExtractedCSI(
        h_f=h_f,
        source="sionna_ofdm_channel",
        source_component="unit_test",
        axes={"B": 0, "Nsc": 1, "K": 2, "Nt": 3},
        shape={"B": 2, "Nsc": 3, "K": 4, "Nt": 5},
        selected_ofdm_symbol=1,
        effective_subcarrier_indices=[1, 2, 3],
        num_users=4,
        num_bs_ant=5,
        num_subcarriers=3,
        project_h_f_assisted=False,
        extracted_h_f_used=True,
        full_native_only=False,
        metadata={
            "original_sionna_h_shape": [2, 4, 1, 1, 5, 2, 7],
            "original_axes": {"batch": 0, "rx": 1},
            "selected_data_symbol": 1,
            "pilot_symbol_indices": [0],
            "effective_subcarrier_ind": [1, 2, 3],
            "extraction_success": True,
            "fallback_reason": "",
        },
    )


def test_as_project_h_f_accepts_extracted_csi() -> None:
    csi = _make_csi(_make_h_f())
    h_f, meta = as_project_h_f(csi)
    assert list(h_f.shape) == [2, 3, 4, 5]
    assert meta["input_type"] == "ExtractedCSI"
    assert meta["csi_interface_used"] is True
    assert meta["project_h_f_assisted"] is False


def test_as_project_h_f_accepts_raw_tensor() -> None:
    h_f = _make_h_f()
    out, meta = as_project_h_f(h_f)
    assert torch.equal(out, h_f.contiguous())
    assert meta["input_type"] == "raw_h_f"
    assert meta["csi_interface_used"] is False


def test_as_project_h_f_accepts_dict_with_h_f() -> None:
    h_f = _make_h_f()
    out, meta = as_project_h_f(
        {
            "h_f": h_f,
            "project_h_f_assisted": True,
            "extracted_h_f_used": False,
            "source": "synthetic_project",
            "source_component": "unit_test_dict",
        }
    )
    assert list(out.shape) == [2, 3, 4, 5]
    assert meta["input_type"] == "dict_with_h_f"
    assert meta["project_h_f_assisted"] is True
    assert meta["source_component"] == "unit_test_dict"


def test_as_project_h_f_wrong_shape_raises_clear_error() -> None:
    h_f = torch.ones(2, 3, 4, dtype=torch.complex64)
    with pytest.raises(ValueError, match="expected_rank_4_h_f_got_3"):
        as_project_h_f(h_f)


def test_require_extracted_csi_rejects_raw_tensor() -> None:
    with pytest.raises(TypeError, match="expected_ExtractedCSI_input_got_Tensor"):
        require_extracted_csi(_make_h_f())


def test_summarize_csi_input_includes_provenance_fields() -> None:
    summary = summarize_csi_input(_make_csi(_make_h_f()))
    assert summary["input_type"] == "ExtractedCSI"
    assert summary["csi_interface_used"] is True
    assert summary["project_h_f_assisted"] is False
    assert summary["extracted_h_f_used"] is True
    assert summary["source"] == "sionna_ofdm_channel"
    assert summary["source_component"] == "unit_test"
    assert summary["tensor_signature"] is not None


def test_compare_csi_inputs_reports_same_shape_and_signature() -> None:
    csi = _make_csi(_make_h_f())
    result = compare_csi_inputs(csi, {"h_f": csi.to_project_h_f(), "source": csi.source, "source_component": csi.source_component})
    assert result["same_shape"] is True
    assert result["same_tensor_signature"] is True
    assert result["same_source"] is True
