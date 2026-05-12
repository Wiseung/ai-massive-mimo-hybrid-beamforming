from __future__ import annotations

import pytest
import torch

from beamforming.utils.precoder_interface import (
    PrecoderOutput,
    as_project_f_f,
    compare_precoder_outputs,
    require_precoder_output,
    summarize_precoder_input,
)


def _make_f_f() -> torch.Tensor:
    real = torch.randn(2, 3, 5, 4, dtype=torch.float32)
    imag = torch.randn(2, 3, 5, 4, dtype=torch.float32)
    f_f = (real + 1j * imag).to(torch.complex64)
    power = (torch.abs(f_f) ** 2).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return f_f / torch.sqrt(power)


def _make_precoder_output(f_f: torch.Tensor) -> PrecoderOutput:
    return PrecoderOutput(
        f_f=f_f,
        source="learned_residual_rzf",
        method="learned_residual_rzf",
        input_csi_summary={"input_type": "ExtractedCSI", "h_f_shape": [2, 3, 4, 5], "source": "sionna_ofdm_channel"},
        axes={"B": 0, "Nsc": 1, "Nt": 2, "K": 3},
        shape={"B": 2, "Nsc": 3, "Nt": 5, "K": 4},
        num_users=4,
        num_bs_ant=5,
        num_subcarriers=3,
        power_normalized=True,
        power_norm={"mean": 1.0},
        teacher_used_during_inference=False,
        project_side_precoder=True,
        sionna_native_precoder=False,
        full_native_only=False,
        metadata={},
    )


def test_as_project_f_f_accepts_precoder_output() -> None:
    precoder = _make_precoder_output(_make_f_f())
    f_f, meta = as_project_f_f(precoder)
    assert list(f_f.shape) == [2, 3, 5, 4]
    assert meta["input_type"] == "PrecoderOutput"
    assert meta["precoder_interface_used"] is True


def test_as_project_f_f_accepts_raw_tensor() -> None:
    f_f = _make_f_f()
    out, meta = as_project_f_f(f_f)
    assert torch.equal(out, f_f.contiguous())
    assert meta["input_type"] == "raw_f_f"
    assert meta["precoder_interface_used"] is False


def test_as_project_f_f_accepts_dict_with_f_f() -> None:
    f_f = _make_f_f()
    out, meta = as_project_f_f({"f_f": f_f, "method": "project_rzf", "project_side_precoder": True})
    assert list(out.shape) == [2, 3, 5, 4]
    assert meta["input_type"] == "dict_with_f_f"
    assert meta["method"] == "project_rzf"


def test_as_project_f_f_wrong_shape_raises_clear_error() -> None:
    f_f = torch.ones(2, 3, 5, dtype=torch.complex64)
    with pytest.raises(ValueError, match="expected_rank_4_f_f_got_3"):
        as_project_f_f(f_f)


def test_require_precoder_output_rejects_raw_tensor() -> None:
    with pytest.raises(TypeError, match="expected_PrecoderOutput_input_got_Tensor"):
        require_precoder_output(_make_f_f())


def test_summarize_precoder_input_contains_provenance_fields() -> None:
    summary = summarize_precoder_input(_make_precoder_output(_make_f_f()))
    assert summary["input_type"] == "PrecoderOutput"
    assert summary["precoder_interface_used"] is True
    assert summary["method"] == "learned_residual_rzf"
    assert summary["teacher_used_during_inference"] is False


def test_compare_precoder_outputs_reports_same_signature() -> None:
    precoder = _make_precoder_output(_make_f_f())
    result = compare_precoder_outputs(precoder, {"f_f": precoder.to_project_f_f(), "source": precoder.source, "method": precoder.method})
    assert result["same_shape"] is True
    assert result["same_tensor_signature"] is True
    assert result["same_method"] is True
    assert result["max_abs_diff"] == pytest.approx(0.0)
