"""Standardized CSI container for project-side beamforming interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


_SUPPORTED_CSI_SOURCES = {
    "sionna_ofdm_channel",
    "synthetic_project",
    "deepmimo_future",
}

_EXPECTED_AXES = {"B": 0, "Nsc": 1, "K": 2, "Nt": 3}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def tensor_signature(tensor: torch.Tensor | None) -> str | None:
    """Return a stable SHA256 signature for a tensor."""
    if tensor is None:
        return None
    tensor_cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor_cpu.numpy().tobytes()).hexdigest()


def _validate_raw_project_h_f_tensor(
    h_f: torch.Tensor,
    *,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate a raw project-side ``H_f`` tensor with shape ``(B,Nsc,K,Nt)``."""

    report: dict[str, Any] = {
        "valid": False,
        "shape": [int(x) for x in h_f.shape],
        "dtype": str(h_f.dtype),
        "is_complex": bool(torch.is_complex(h_f)),
        "all_finite": False,
        "fallback_reason": "",
    }

    def _fail(exc_type: type[Exception], reason: str) -> dict[str, Any]:
        report["fallback_reason"] = reason
        if raise_on_error:
            raise exc_type(reason)
        return report

    if h_f.ndim != 4:
        return _fail(ValueError, f"expected_rank_4_h_f_got_{h_f.ndim}")
    if not torch.is_complex(h_f):
        return _fail(TypeError, "h_f_is_not_complex")
    all_finite = bool(torch.isfinite(h_f.real).all() and torch.isfinite(h_f.imag).all())
    report["all_finite"] = all_finite
    if not all_finite:
        return _fail(ValueError, "h_f_contains_non_finite_values")
    report["valid"] = True
    return report


def as_project_h_f(
    input_obj: ExtractedCSI | torch.Tensor | dict[str, Any],
    *,
    validate: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize supported CSI inputs to a project ``H_f`` tensor.

    Supported inputs:
    - ``ExtractedCSI``
    - raw complex ``torch.Tensor`` with shape ``(B,Nsc,K,Nt)``
    - ``dict`` containing ``"h_f"``, optionally with provenance metadata
    """

    if isinstance(input_obj, ExtractedCSI):
        h_f = input_obj.to_project_h_f() if validate else input_obj.h_f.contiguous()
        meta = {
            "input_type": "ExtractedCSI",
            "csi_interface_used": True,
            "project_h_f_assisted": bool(input_obj.project_h_f_assisted),
            "extracted_h_f_used": bool(input_obj.extracted_h_f_used),
            "full_native_only": bool(input_obj.full_native_only),
            "source": input_obj.source,
            "source_component": input_obj.source_component,
            "validation": input_obj.validate(raise_on_error=False),
            "tensor_signature": tensor_signature(h_f),
        }
        return h_f, meta

    if isinstance(input_obj, dict):
        if "h_f" not in input_obj:
            raise KeyError("dict_input_missing_h_f")
        h_f, child_meta = as_project_h_f(input_obj["h_f"], validate=validate)
        meta = {
            **child_meta,
            "input_type": "dict_with_h_f",
            "project_h_f_assisted": input_obj.get("project_h_f_assisted", child_meta.get("project_h_f_assisted")),
            "extracted_h_f_used": input_obj.get("extracted_h_f_used", child_meta.get("extracted_h_f_used")),
            "full_native_only": input_obj.get("full_native_only", child_meta.get("full_native_only")),
            "source": input_obj.get("source", child_meta.get("source")),
            "source_component": input_obj.get("source_component", child_meta.get("source_component")),
            "dict_keys": sorted(str(key) for key in input_obj.keys()),
        }
        if "csi_interface_used" in input_obj:
            meta["csi_interface_used"] = bool(input_obj["csi_interface_used"])
        return h_f, meta

    if isinstance(input_obj, torch.Tensor):
        h_f = input_obj.contiguous()
        validation = _validate_raw_project_h_f_tensor(h_f, raise_on_error=validate)
        meta = {
            "input_type": "raw_h_f",
            "csi_interface_used": False,
            "project_h_f_assisted": None,
            "extracted_h_f_used": None,
            "full_native_only": False,
            "source": None,
            "source_component": None,
            "validation": validation,
            "tensor_signature": tensor_signature(h_f),
        }
        return h_f, meta

    raise TypeError(f"unsupported_csi_input_type_{type(input_obj).__name__}")


def require_extracted_csi(input_obj: Any) -> ExtractedCSI:
    """Require an ``ExtractedCSI`` object and raise a clear error otherwise."""
    if not isinstance(input_obj, ExtractedCSI):
        raise TypeError(f"expected_ExtractedCSI_input_got_{type(input_obj).__name__}")
    return input_obj


def summarize_csi_input(input_obj: ExtractedCSI | torch.Tensor | dict[str, Any]) -> dict[str, Any]:
    """Return a serializable summary for a CSI input object."""
    h_f, meta = as_project_h_f(input_obj, validate=False)
    return {
        "input_type": meta["input_type"],
        "h_f_shape": [int(x) for x in h_f.shape],
        "csi_interface_used": meta.get("csi_interface_used"),
        "project_h_f_assisted": meta.get("project_h_f_assisted"),
        "extracted_h_f_used": meta.get("extracted_h_f_used"),
        "full_native_only": meta.get("full_native_only"),
        "source": meta.get("source"),
        "source_component": meta.get("source_component"),
        "tensor_signature": meta.get("tensor_signature"),
    }


def compare_csi_inputs(
    a: ExtractedCSI | torch.Tensor | dict[str, Any],
    b: ExtractedCSI | torch.Tensor | dict[str, Any],
) -> dict[str, Any]:
    """Compare two CSI inputs for shape, signature, and provenance consistency."""
    h_f_a, meta_a = as_project_h_f(a, validate=False)
    h_f_b, meta_b = as_project_h_f(b, validate=False)
    return {
        "same_shape": list(h_f_a.shape) == list(h_f_b.shape),
        "same_tensor_signature": tensor_signature(h_f_a) == tensor_signature(h_f_b),
        "same_input_type": meta_a.get("input_type") == meta_b.get("input_type"),
        "same_csi_interface_used": meta_a.get("csi_interface_used") == meta_b.get("csi_interface_used"),
        "same_project_h_f_assisted": meta_a.get("project_h_f_assisted") == meta_b.get("project_h_f_assisted"),
        "same_extracted_h_f_used": meta_a.get("extracted_h_f_used") == meta_b.get("extracted_h_f_used"),
        "same_source": meta_a.get("source") == meta_b.get("source"),
        "same_source_component": meta_a.get("source_component") == meta_b.get("source_component"),
        "a": summarize_csi_input(a),
        "b": summarize_csi_input(b),
    }


@dataclass
class ExtractedCSI:
    """Standardized project-side CSI object with provenance metadata.

    The normalized project tensor convention is:
    ``h_f.shape == (B, Nsc, K, Nt)``.
    """

    h_f: torch.Tensor
    source: str
    source_component: str
    axes: dict[str, int]
    shape: dict[str, int]
    selected_ofdm_symbol: int
    effective_subcarrier_indices: list[int]
    num_users: int
    num_bs_ant: int
    num_subcarriers: int
    project_h_f_assisted: bool
    extracted_h_f_used: bool
    full_native_only: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, raise_on_error: bool = True) -> dict[str, Any]:
        """Validate shape, dtype, finiteness, and provenance fields."""

        report: dict[str, Any] = {
            "valid": False,
            "shape": [int(x) for x in self.h_f.shape],
            "dtype": str(self.h_f.dtype),
            "is_complex": bool(torch.is_complex(self.h_f)),
            "all_finite": False,
            "norm_mean": None,
            "norm_std": None,
            "subcarrier_norm_std_mean": None,
            "fallback_reason": "",
        }

        def _fail(exc_type: type[Exception], reason: str) -> dict[str, Any]:
            report["fallback_reason"] = reason
            if raise_on_error:
                raise exc_type(reason)
            return report

        if self.source not in _SUPPORTED_CSI_SOURCES:
            return _fail(ValueError, f"unsupported_csi_source_{self.source}")
        if self.axes != _EXPECTED_AXES:
            return _fail(ValueError, f"expected_axes_{_EXPECTED_AXES}_got_{self.axes}")
        if self.h_f.ndim != 4:
            return _fail(ValueError, f"expected_rank_4_h_f_got_{self.h_f.ndim}")
        if not torch.is_complex(self.h_f):
            return _fail(TypeError, "h_f_is_not_complex")
        if self.shape != {
            "B": int(self.h_f.size(0)),
            "Nsc": int(self.h_f.size(1)),
            "K": int(self.h_f.size(2)),
            "Nt": int(self.h_f.size(3)),
        }:
            return _fail(ValueError, f"shape_metadata_mismatch_tensor_shape_{self.shape}")
        if self.num_users != int(self.h_f.size(2)):
            return _fail(ValueError, "num_users_does_not_match_h_f_shape")
        if self.num_bs_ant != int(self.h_f.size(3)):
            return _fail(ValueError, "num_bs_ant_does_not_match_h_f_shape")
        if self.num_subcarriers != int(self.h_f.size(1)):
            return _fail(ValueError, "num_subcarriers_does_not_match_h_f_shape")
        if len(self.effective_subcarrier_indices) != int(self.h_f.size(1)):
            return _fail(ValueError, "effective_subcarrier_indices_length_mismatch")
        if not isinstance(self.selected_ofdm_symbol, int):
            return _fail(TypeError, "selected_ofdm_symbol_must_be_int")

        all_finite = bool(torch.isfinite(self.h_f.real).all() and torch.isfinite(self.h_f.imag).all())
        report["all_finite"] = all_finite
        if not all_finite:
            return _fail(ValueError, "h_f_contains_non_finite_values")

        norms = torch.linalg.vector_norm(self.h_f, dim=(-2, -1))
        report["norm_mean"] = float(norms.mean().item())
        report["norm_std"] = float(norms.std(unbiased=False).item())
        report["subcarrier_norm_std_mean"] = float(norms.std(dim=1, unbiased=False).mean().item())
        report["valid"] = True
        return report

    def to_project_h_f(self) -> torch.Tensor:
        self.validate()
        return self.h_f.contiguous()

    def summary_dict(self) -> dict[str, Any]:
        validation = self.validate(raise_on_error=False)
        return {
            "source": self.source,
            "source_component": self.source_component,
            "axes": dict(self.axes),
            "shape": dict(self.shape),
            "h_f_shape": [int(x) for x in self.h_f.shape],
            "dtype": str(self.h_f.dtype),
            "selected_ofdm_symbol": int(self.selected_ofdm_symbol),
            "effective_subcarrier_indices": [int(x) for x in self.effective_subcarrier_indices],
            "num_users": int(self.num_users),
            "num_bs_ant": int(self.num_bs_ant),
            "num_subcarriers": int(self.num_subcarriers),
            "project_h_f_assisted": bool(self.project_h_f_assisted),
            "extracted_h_f_used": bool(self.extracted_h_f_used),
            "full_native_only": bool(self.full_native_only),
            "validation": validation,
            "metadata": _json_safe(self.metadata),
        }

    def save_summary_json(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.summary_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class SharedSionnaOFDMBatch:
    """Deterministic shared batch for raw-H and CSI-backed path equivalence tests."""

    bits: torch.Tensor
    symbols: torch.Tensor
    resource_grid: Any
    stream_management: Any
    sionna_channel_tensor: torch.Tensor
    extracted_h_f: torch.Tensor
    csi: ExtractedCSI
    rx_noise_grid: torch.Tensor
    noise_var: float
    snr_db: float
    seed: int
    selected_ofdm_symbol: int
    effective_subcarrier_indices: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "batch_size": int(self.bits.size(0)),
            "snr_db": float(self.snr_db),
            "selected_ofdm_symbol": int(self.selected_ofdm_symbol),
            "effective_subcarrier_indices": [int(x) for x in self.effective_subcarrier_indices],
            "original_sionna_h_shape": [int(x) for x in self.sionna_channel_tensor.shape],
            "extracted_h_f_shape": [int(x) for x in self.extracted_h_f.shape],
            "bits_signature": tensor_signature(self.bits),
            "symbols_signature": tensor_signature(self.symbols),
            "channel_tensor_signature": tensor_signature(self.sionna_channel_tensor),
            "extracted_h_f_signature": tensor_signature(self.extracted_h_f),
            "rx_noise_grid_signature": tensor_signature(self.rx_noise_grid),
            "noise_var": float(self.noise_var),
            "metadata": _json_safe(self.metadata),
        }
