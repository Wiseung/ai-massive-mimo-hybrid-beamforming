from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from beamforming.utils.precoder_interface import PrecoderOutput


def _make_f_f() -> torch.Tensor:
    real = torch.randn(2, 3, 5, 4, dtype=torch.float32)
    imag = torch.randn(2, 3, 5, 4, dtype=torch.float32)
    f_f = (real + 1j * imag).to(torch.complex64)
    power = (torch.abs(f_f) ** 2).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return f_f / torch.sqrt(power)


def _make_precoder_output(f_f: torch.Tensor) -> PrecoderOutput:
    return PrecoderOutput(
        f_f=f_f,
        source="project_rzf",
        method="project_rzf",
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
        metadata={
            "input_csi_source": "sionna_ofdm_channel",
            "input_h_f_shape": [2, 3, 4, 5],
            "checkpoint_path": None,
            "skipped_missing_checkpoint": False,
            "teacher_used_during_inference": False,
            "fallback_reason": "",
        },
    )


def test_precoder_interface_valid_shape_passes() -> None:
    precoder = _make_precoder_output(_make_f_f())
    report = precoder.validate(raise_on_error=False)
    assert report["valid"] is True
    assert report["all_finite"] is True
    assert report["power_summary"] is not None


def test_precoder_interface_wrong_shape_raises_clear_error() -> None:
    f_f = torch.ones(2, 3, 5, dtype=torch.complex64)
    precoder = _make_precoder_output(_make_f_f())
    precoder.f_f = f_f
    with pytest.raises(ValueError, match="expected_rank_4_f_f_got_3"):
        precoder.validate()


def test_precoder_interface_no_nan_or_inf() -> None:
    precoder = _make_precoder_output(_make_f_f())
    assert precoder.validate(raise_on_error=False)["all_finite"] is True


def test_precoder_interface_teacher_flag_tracked() -> None:
    precoder = _make_precoder_output(_make_f_f())
    assert precoder.teacher_used_during_inference is False
    summary = precoder.summary_dict()
    assert summary["teacher_used_during_inference"] is False


def test_precoder_interface_summary_dict_contains_provenance_fields() -> None:
    precoder = _make_precoder_output(_make_f_f())
    summary = precoder.summary_dict()
    assert summary["source"] == "project_rzf"
    assert summary["method"] == "project_rzf"
    assert summary["input_csi_summary"]["source"] == "sionna_ofdm_channel"
    assert "metadata" in summary


def test_precoder_interface_save_summary_json(tmp_path: Path) -> None:
    precoder = _make_precoder_output(_make_f_f())
    out_path = tmp_path / "precoder_summary.json"
    precoder.save_summary_json(out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["f_f_shape"] == [2, 3, 5, 4]
