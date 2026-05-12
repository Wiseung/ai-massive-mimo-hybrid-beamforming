"""Standardized precoder-output container for project/native beamforming paths."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch

from beamforming.utils.csi_interface import ExtractedCSI, _json_safe, summarize_csi_input, tensor_signature


_SUPPORTED_PRECODER_SOURCES = {
    "project_rzf",
    "project_wmmse_iter_5",
    "learned_residual_rzf",
    "learned_residual_wmmse_distill",
    "sionna_rzf_precoder",
    "sionna_rzf_future",
}

_EXPECTED_AXES = {"B": 0, "Nsc": 1, "Nt": 2, "K": 3}


def _normalize_power_norm(value: Any) -> float | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return {"values": value.detach().cpu().tolist()}
    if isinstance(value, (int, float)):
        return float(value)
    return {"value": _json_safe(value)}


def _validate_raw_project_f_f_tensor(
    f_f: torch.Tensor,
    *,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "valid": False,
        "shape": [int(x) for x in f_f.shape],
        "dtype": str(f_f.dtype),
        "is_complex": bool(torch.is_complex(f_f)),
        "all_finite": False,
        "fallback_reason": "",
    }

    def _fail(exc_type: type[Exception], reason: str) -> dict[str, Any]:
        report["fallback_reason"] = reason
        if raise_on_error:
            raise exc_type(reason)
        return report

    if f_f.ndim != 4:
        return _fail(ValueError, f"expected_rank_4_f_f_got_{f_f.ndim}")
    if not torch.is_complex(f_f):
        return _fail(TypeError, "f_f_is_not_complex")
    all_finite = bool(torch.isfinite(f_f.real).all() and torch.isfinite(f_f.imag).all())
    report["all_finite"] = all_finite
    if not all_finite:
        return _fail(ValueError, "f_f_contains_non_finite_values")
    report["valid"] = True
    return report


def _power_summary_from_tensor(f_f: torch.Tensor) -> dict[str, float]:
    power = (torch.abs(f_f) ** 2).sum(dim=(-2, -1))
    return {
        "mean": float(power.mean().item()),
        "std": float(power.std(unbiased=False).item()),
        "max_abs_deviation_from_one": float(torch.abs(power - 1.0).max().item()),
    }


def _max_abs_complex_diff(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if list(a.shape) != list(b.shape):
        return None
    return float(torch.max(torch.abs(a - b)).item())


@dataclass
class PrecoderOutput:
    """Standardized project-side precoder output with provenance metadata.

    The normalized project tensor convention is:
    ``f_f.shape == (B, Nsc, Nt, K)``.
    """

    f_f: torch.Tensor
    source: str
    method: str
    input_csi_summary: dict[str, Any]
    axes: dict[str, int]
    shape: dict[str, int]
    num_users: int
    num_bs_ant: int
    num_subcarriers: int
    power_normalized: bool
    power_norm: float | dict[str, Any] | None
    teacher_used_during_inference: bool
    project_side_precoder: bool
    sionna_native_precoder: bool
    full_native_only: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, raise_on_error: bool = True) -> dict[str, Any]:
        report: dict[str, Any] = {
            "valid": False,
            "shape": [int(x) for x in self.f_f.shape],
            "dtype": str(self.f_f.dtype),
            "is_complex": bool(torch.is_complex(self.f_f)),
            "all_finite": False,
            "power_summary": None,
            "fallback_reason": "",
        }

        def _fail(exc_type: type[Exception], reason: str) -> dict[str, Any]:
            report["fallback_reason"] = reason
            if raise_on_error:
                raise exc_type(reason)
            return report

        if self.source not in _SUPPORTED_PRECODER_SOURCES:
            return _fail(ValueError, f"unsupported_precoder_source_{self.source}")
        if self.axes != _EXPECTED_AXES:
            return _fail(ValueError, f"expected_axes_{_EXPECTED_AXES}_got_{self.axes}")
        if self.f_f.ndim != 4:
            return _fail(ValueError, f"expected_rank_4_f_f_got_{self.f_f.ndim}")
        if not torch.is_complex(self.f_f):
            return _fail(TypeError, "f_f_is_not_complex")
        expected_shape = {
            "B": int(self.f_f.size(0)),
            "Nsc": int(self.f_f.size(1)),
            "Nt": int(self.f_f.size(2)),
            "K": int(self.f_f.size(3)),
        }
        if self.shape != expected_shape:
            return _fail(ValueError, f"shape_metadata_mismatch_tensor_shape_{self.shape}")
        if self.num_bs_ant != int(self.f_f.size(2)):
            return _fail(ValueError, "num_bs_ant_does_not_match_f_f_shape")
        if self.num_users != int(self.f_f.size(3)):
            return _fail(ValueError, "num_users_does_not_match_f_f_shape")
        if self.num_subcarriers != int(self.f_f.size(1)):
            return _fail(ValueError, "num_subcarriers_does_not_match_f_f_shape")
        if not isinstance(self.input_csi_summary, dict):
            return _fail(TypeError, "input_csi_summary_must_be_dict")

        all_finite = bool(torch.isfinite(self.f_f.real).all() and torch.isfinite(self.f_f.imag).all())
        report["all_finite"] = all_finite
        if not all_finite:
            return _fail(ValueError, "f_f_contains_non_finite_values")

        report["power_summary"] = _power_summary_from_tensor(self.f_f)
        report["valid"] = True
        return report

    def to_project_f_f(self) -> torch.Tensor:
        self.validate()
        return self.f_f.contiguous()

    def summary_dict(self) -> dict[str, Any]:
        validation = self.validate(raise_on_error=False)
        return {
            "source": self.source,
            "method": self.method,
            "input_csi_summary": _json_safe(self.input_csi_summary),
            "axes": dict(self.axes),
            "shape": dict(self.shape),
            "f_f_shape": [int(x) for x in self.f_f.shape],
            "dtype": str(self.f_f.dtype),
            "num_users": int(self.num_users),
            "num_bs_ant": int(self.num_bs_ant),
            "num_subcarriers": int(self.num_subcarriers),
            "power_normalized": bool(self.power_normalized),
            "power_norm": _json_safe(self.power_norm),
            "teacher_used_during_inference": bool(self.teacher_used_during_inference),
            "project_side_precoder": bool(self.project_side_precoder),
            "sionna_native_precoder": bool(self.sionna_native_precoder),
            "full_native_only": bool(self.full_native_only),
            "validation": validation,
            "metadata": _json_safe(self.metadata),
        }

    def save_summary_json(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.summary_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def as_project_f_f(
    input_obj: PrecoderOutput | torch.Tensor | dict[str, Any],
    *,
    validate: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize supported precoder inputs to a project ``F_f`` tensor."""

    if isinstance(input_obj, PrecoderOutput):
        f_f = input_obj.to_project_f_f() if validate else input_obj.f_f.contiguous()
        meta = {
            "input_type": "PrecoderOutput",
            "precoder_interface_used": True,
            "method": input_obj.method,
            "source": input_obj.source,
            "project_side_precoder": bool(input_obj.project_side_precoder),
            "sionna_native_precoder": bool(input_obj.sionna_native_precoder),
            "full_native_only": bool(input_obj.full_native_only),
            "teacher_used_during_inference": bool(input_obj.teacher_used_during_inference),
            "power_normalized": bool(input_obj.power_normalized),
            "power_norm": _json_safe(input_obj.power_norm),
            "validation": input_obj.validate(raise_on_error=False),
            "tensor_signature": tensor_signature(f_f),
        }
        return f_f, meta

    if isinstance(input_obj, dict):
        if "f_f" not in input_obj:
            raise KeyError("dict_input_missing_f_f")
        f_f, child_meta = as_project_f_f(input_obj["f_f"], validate=validate)
        meta = {
            **child_meta,
            "input_type": "dict_with_f_f",
            "method": input_obj.get("method", child_meta.get("method")),
            "source": input_obj.get("source", child_meta.get("source")),
            "project_side_precoder": input_obj.get("project_side_precoder", child_meta.get("project_side_precoder")),
            "sionna_native_precoder": input_obj.get("sionna_native_precoder", child_meta.get("sionna_native_precoder")),
            "full_native_only": input_obj.get("full_native_only", child_meta.get("full_native_only")),
            "teacher_used_during_inference": input_obj.get(
                "teacher_used_during_inference",
                child_meta.get("teacher_used_during_inference"),
            ),
            "power_normalized": input_obj.get("power_normalized", child_meta.get("power_normalized")),
            "power_norm": input_obj.get("power_norm", child_meta.get("power_norm")),
            "dict_keys": sorted(str(key) for key in input_obj.keys()),
        }
        if "precoder_interface_used" in input_obj:
            meta["precoder_interface_used"] = bool(input_obj["precoder_interface_used"])
        return f_f, meta

    if isinstance(input_obj, torch.Tensor):
        f_f = input_obj.contiguous()
        validation = _validate_raw_project_f_f_tensor(f_f, raise_on_error=validate)
        meta = {
            "input_type": "raw_f_f",
            "precoder_interface_used": False,
            "method": None,
            "source": None,
            "project_side_precoder": None,
            "sionna_native_precoder": None,
            "full_native_only": False,
            "teacher_used_during_inference": False,
            "power_normalized": None,
            "power_norm": _power_summary_from_tensor(f_f) if validation.get("valid") else None,
            "validation": validation,
            "tensor_signature": tensor_signature(f_f),
        }
        return f_f, meta

    raise TypeError(f"unsupported_precoder_input_type_{type(input_obj).__name__}")


def require_precoder_output(input_obj: Any) -> PrecoderOutput:
    if not isinstance(input_obj, PrecoderOutput):
        raise TypeError(f"expected_PrecoderOutput_input_got_{type(input_obj).__name__}")
    return input_obj


def summarize_precoder_input(input_obj: PrecoderOutput | torch.Tensor | dict[str, Any]) -> dict[str, Any]:
    f_f, meta = as_project_f_f(input_obj, validate=False)
    return {
        "input_type": meta["input_type"],
        "f_f_shape": [int(x) for x in f_f.shape],
        "precoder_interface_used": meta.get("precoder_interface_used"),
        "method": meta.get("method"),
        "source": meta.get("source"),
        "project_side_precoder": meta.get("project_side_precoder"),
        "sionna_native_precoder": meta.get("sionna_native_precoder"),
        "full_native_only": meta.get("full_native_only"),
        "teacher_used_during_inference": meta.get("teacher_used_during_inference"),
        "power_normalized": meta.get("power_normalized"),
        "power_norm": meta.get("power_norm"),
        "tensor_signature": meta.get("tensor_signature"),
    }


def compare_precoder_outputs(
    a: PrecoderOutput | torch.Tensor | dict[str, Any],
    b: PrecoderOutput | torch.Tensor | dict[str, Any],
) -> dict[str, Any]:
    f_f_a, meta_a = as_project_f_f(a, validate=False)
    f_f_b, meta_b = as_project_f_f(b, validate=False)
    return {
        "same_shape": list(f_f_a.shape) == list(f_f_b.shape),
        "same_tensor_signature": tensor_signature(f_f_a) == tensor_signature(f_f_b),
        "max_abs_diff": _max_abs_complex_diff(f_f_a, f_f_b),
        "same_input_type": meta_a.get("input_type") == meta_b.get("input_type"),
        "same_precoder_interface_used": meta_a.get("precoder_interface_used") == meta_b.get("precoder_interface_used"),
        "same_method": meta_a.get("method") == meta_b.get("method"),
        "same_source": meta_a.get("source") == meta_b.get("source"),
        "same_project_side_precoder": meta_a.get("project_side_precoder") == meta_b.get("project_side_precoder"),
        "same_teacher_used_during_inference": meta_a.get("teacher_used_during_inference")
        == meta_b.get("teacher_used_during_inference"),
        "a": summarize_precoder_input(a),
        "b": summarize_precoder_input(b),
    }


def build_precoder_output(
    *,
    f_f: torch.Tensor,
    source: str,
    method: str,
    input_csi: ExtractedCSI | torch.Tensor | dict[str, Any],
    project_side_precoder: bool,
    sionna_native_precoder: bool,
    teacher_used_during_inference: bool,
    power_normalized: bool,
    checkpoint_path: str | None = None,
    skipped_missing_checkpoint: bool = False,
    fallback_reason: str = "",
    full_native_only: bool = False,
    metadata: dict[str, Any] | None = None,
) -> PrecoderOutput:
    csi_summary = summarize_csi_input(input_csi) if not isinstance(input_csi, dict) or "h_f" in input_csi else dict(input_csi)
    combined_metadata = dict(metadata or {})
    combined_metadata.setdefault("input_csi_source", csi_summary.get("source"))
    combined_metadata.setdefault("input_h_f_shape", csi_summary.get("h_f_shape"))
    combined_metadata.setdefault("checkpoint_path", checkpoint_path)
    combined_metadata.setdefault("skipped_missing_checkpoint", bool(skipped_missing_checkpoint))
    combined_metadata.setdefault("teacher_used_during_inference", bool(teacher_used_during_inference))
    combined_metadata.setdefault("fallback_reason", fallback_reason)
    return PrecoderOutput(
        f_f=f_f.contiguous(),
        source=source,
        method=method,
        input_csi_summary=csi_summary,
        axes=dict(_EXPECTED_AXES),
        shape={"B": int(f_f.size(0)), "Nsc": int(f_f.size(1)), "Nt": int(f_f.size(2)), "K": int(f_f.size(3))},
        num_users=int(f_f.size(3)),
        num_bs_ant=int(f_f.size(2)),
        num_subcarriers=int(f_f.size(1)),
        power_normalized=bool(power_normalized),
        power_norm=_power_summary_from_tensor(f_f),
        teacher_used_during_inference=bool(teacher_used_during_inference),
        project_side_precoder=bool(project_side_precoder),
        sionna_native_precoder=bool(sionna_native_precoder),
        full_native_only=bool(full_native_only),
        metadata=combined_metadata,
    )


@dataclass
class SharedPrecoderMethodArtifacts:
    """One deterministic raw/PrecoderOutput pair for a single method."""

    method: str
    method_type: str
    raw_f_f: torch.Tensor
    precoder_output: PrecoderOutput
    checkpoint_path: str | None
    teacher_used_during_inference: bool
    runtime_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        comparison = compare_precoder_outputs(self.raw_f_f, self.precoder_output)
        return {
            "method": self.method,
            "method_type": self.method_type,
            "checkpoint_path": self.checkpoint_path,
            "teacher_used_during_inference": bool(self.teacher_used_during_inference),
            "runtime_ms": float(self.runtime_ms),
            "raw_f_f_summary": summarize_precoder_input(self.raw_f_f),
            "precoder_output_summary": self.precoder_output.summary_dict(),
            "max_abs_diff_raw_vs_precoder_f_f": comparison["max_abs_diff"],
            "comparison": comparison,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class SharedPrecoderOutputBatch:
    """Deterministic shared batch for raw-F_f vs PrecoderOutput equivalence tests."""

    csi: ExtractedCSI
    raw_h_f: torch.Tensor
    bits: torch.Tensor
    symbols: torch.Tensor
    resource_grid: Any
    stream_management: Any
    h_full: torch.Tensor
    rx_noise_grid: torch.Tensor | None
    noise_var: float
    snr_db: float
    seed: int
    selected_ofdm_symbol: int
    effective_subcarrier_indices: list[int]
    receiver_config: dict[str, Any]
    method_artifacts: dict[str, SharedPrecoderMethodArtifacts]
    skipped_methods: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    native_context: Any | None = None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "batch_size": int(self.bits.size(0)),
            "snr_db": float(self.snr_db),
            "noise_var": float(self.noise_var),
            "selected_ofdm_symbol": int(self.selected_ofdm_symbol),
            "effective_subcarrier_indices": [int(x) for x in self.effective_subcarrier_indices],
            "raw_h_f_shape": [int(x) for x in self.raw_h_f.shape],
            "h_full_shape": [int(x) for x in self.h_full.shape],
            "raw_h_f_signature": tensor_signature(self.raw_h_f),
            "h_full_signature": tensor_signature(self.h_full),
            "bits_signature": tensor_signature(self.bits),
            "symbols_signature": tensor_signature(self.symbols),
            "rx_noise_grid_signature": tensor_signature(self.rx_noise_grid),
            "csi_summary": self.csi.summary_dict(),
            "receiver_config": _json_safe(self.receiver_config),
            "method_artifacts": {
                method: item.summary_dict() for method, item in self.method_artifacts.items()
            },
            "skipped_methods": list(self.skipped_methods),
            "metadata": _json_safe(self.metadata),
        }


def create_shared_precoder_output_batch(
    *,
    batch_size: int,
    num_subcarriers: int,
    num_users: int,
    num_bs_ant: int,
    snr_db: float,
    device: torch.device,
    repo_root: Path,
    seed: int = 0,
    methods: list[tuple[str, str]] | None = None,
) -> SharedPrecoderOutputBatch:
    """Create one deterministic batch shared by raw-F_f and PrecoderOutput paths."""

    from beamforming.utils.sionna_native_beamforming_chain import (
        compute_project_precoder_per_subcarrier,
        summarize_receiver_config,
    )
    from beamforming.utils.sionna_native_learned_beamforming import (
        build_native_receiver_context,
        default_checkpoint_path,
        generate_shared_sionna_channel_bundle,
        infer_learned_precoder,
        load_learned_beamformer_checkpoint,
    )

    method_specs = methods or [
        ("project_rzf", "analytic"),
        ("project_wmmse_iter_5", "analytic"),
        ("learned_residual_rzf", "learned"),
        ("learned_residual_wmmse_distill", "learned"),
    ]
    noise_var = float(10.0 ** (-float(snr_db) / 10.0))
    channel_bundle = generate_shared_sionna_channel_bundle(
        batch_size=batch_size,
        num_subcarriers=num_subcarriers,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        noise_var=noise_var,
        device=device,
        seed=seed,
    )
    native_context = build_native_receiver_context(
        batch_size=batch_size,
        num_subcarriers=num_subcarriers,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        snr_db=snr_db,
        device=device,
        channel_bundle=channel_bundle,
    )
    csi = native_context.csi
    if csi is None:
        raise RuntimeError("shared_precoder_output_batch_requires_extracted_csi")
    receiver_config = summarize_receiver_config(
        native_context.resource_grid,
        native_context.stream_management,
        selected_ofdm_symbol=csi.selected_ofdm_symbol,
        effective_subcarrier_indices=csi.effective_subcarrier_indices,
    )
    method_artifacts: dict[str, SharedPrecoderMethodArtifacts] = {}
    skipped_methods: list[str] = []
    snr_tensor = torch.full((batch_size,), float(snr_db), dtype=torch.float32, device=device)

    for method, method_type in method_specs:
        checkpoint_path: str | None = None
        teacher_flag = False
        runtime_ms = 0.0
        raw_f_f: torch.Tensor
        precoder_output: PrecoderOutput
        metadata: dict[str, Any] = {
            "seed": int(seed),
            "batch_size": int(batch_size),
            "snr_db": float(snr_db),
            "method": method,
            "receiver_config": receiver_config,
            "input_csi_tensor_signature": tensor_signature(csi.to_project_h_f()),
        }
        if method_type == "analytic":
            raw_f_f = compute_project_precoder_per_subcarrier(
                method.removeprefix("project_"),
                csi,
                native_context.noise_var,
                return_precoder_output=False,
            )
            precoder_output = compute_project_precoder_per_subcarrier(
                method.removeprefix("project_"),
                csi,
                native_context.noise_var,
                return_precoder_output=True,
            )
        else:
            ckpt = default_checkpoint_path(method, repo_root)
            checkpoint_path = str(ckpt)
            if not ckpt.exists():
                skipped_methods.append(method)
                continue
            bundle = load_learned_beamformer_checkpoint(ckpt, device, method_name=method)
            raw_precoder, infer_meta, runtime_ms = infer_learned_precoder(
                bundle,
                csi,
                snr_tensor,
                native_receiver_path=True,
                return_precoder_output=False,
            )
            container_precoder, _, _ = infer_learned_precoder(
                bundle,
                csi,
                snr_tensor,
                native_receiver_path=True,
                return_precoder_output=True,
            )
            raw_f_f = raw_precoder
            precoder_output = container_precoder
            teacher_flag = bool(infer_meta.get("teacher_used_during_inference", False))
            metadata["inference_inputs"] = infer_meta.get("inference_inputs")
        comparison = compare_precoder_outputs(raw_f_f, precoder_output)
        metadata.update(
            {
                "checkpoint_path": checkpoint_path,
                "teacher_used_during_inference": bool(teacher_flag),
                "raw_f_f_shape": [int(x) for x in raw_f_f.shape],
                "precoder_output_f_f_shape": [int(x) for x in precoder_output.f_f.shape],
                "raw_f_f_tensor_signature": tensor_signature(raw_f_f),
                "precoder_output_tensor_signature": tensor_signature(precoder_output.f_f),
                "max_abs_diff_raw_vs_precoder_f_f": comparison["max_abs_diff"],
            }
        )
        method_artifacts[method] = SharedPrecoderMethodArtifacts(
            method=method,
            method_type=method_type,
            raw_f_f=raw_f_f.contiguous(),
            precoder_output=precoder_output,
            checkpoint_path=checkpoint_path,
            teacher_used_during_inference=bool(teacher_flag),
            runtime_ms=float(runtime_ms),
            metadata=metadata,
        )

    return SharedPrecoderOutputBatch(
        csi=csi,
        raw_h_f=native_context.h_f.contiguous(),
        bits=native_context.bits.contiguous(),
        symbols=native_context.stream_symbols.contiguous(),
        resource_grid=native_context.resource_grid,
        stream_management=native_context.stream_management,
        h_full=native_context.h_full.contiguous(),
        rx_noise_grid=native_context.context_meta.get("shared_rx_noise_grid"),
        noise_var=float(native_context.noise_var),
        snr_db=float(native_context.snr_db),
        seed=int(seed),
        selected_ofdm_symbol=int(csi.selected_ofdm_symbol),
        effective_subcarrier_indices=[int(x) for x in csi.effective_subcarrier_indices],
        receiver_config=receiver_config,
        method_artifacts=method_artifacts,
        skipped_methods=skipped_methods,
        metadata={
            "seed": int(seed),
            "batch_size": int(batch_size),
            "snr_db": float(snr_db),
            "original_sionna_h_shape": [int(x) for x in native_context.h_full.shape],
            "extracted_h_f_shape": [int(x) for x in native_context.h_f.shape],
        },
        native_context=native_context,
    )
