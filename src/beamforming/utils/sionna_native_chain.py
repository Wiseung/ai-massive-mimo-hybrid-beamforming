"""Helpers for optional Sionna-native OFDM chain auditing and smoke runs."""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COMPONENT_SPECS: dict[str, list[str]] = {
    "ResourceGrid": ["sionna.phy.ofdm"],
    "ResourceGridMapper": ["sionna.phy.ofdm"],
    "ResourceGridDemapper": ["sionna.phy.ofdm"],
    "OFDMChannel": ["sionna.phy.channel", "sionna.phy.ofdm"],
    "ApplyOFDMChannel": ["sionna.phy.channel", "sionna.phy.ofdm"],
    "LSChannelEstimator": ["sionna.phy.ofdm"],
    "LMMSEEqualizer": ["sionna.phy.ofdm"],
    "RemoveNulledSubcarriers": ["sionna.phy.ofdm"],
    "Mapper": ["sionna.phy.mapping"],
    "Demapper": ["sionna.phy.mapping"],
    "BinarySource": ["sionna.phy.mapping"],
    "StreamManagement": ["sionna.phy.mimo"],
    "RayleighBlockFading": ["sionna.phy.channel"],
}


@dataclass
class ComponentProbeResult:
    name: str
    import_ok: bool
    module_path: str | None
    constructor_signature: str | None
    callable_probe_ok: bool
    notes: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "import_ok": self.import_ok,
            "module_path": self.module_path,
            "constructor_signature": self.constructor_signature,
            "minimal_callable_example_available": self.callable_probe_ok,
            "notes": self.notes,
            "error": self.error,
        }


def load_component(name: str) -> tuple[Any | None, str | None, str | None]:
    """Load a Sionna component from the preferred module list."""
    for module_path in COMPONENT_SPECS.get(name, []):
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, name):
                return getattr(module, name), module_path, None
        except Exception as exc:  # pragma: no cover - optional dependency path
            last_error = f"{type(exc).__name__}: {exc}"
            continue
    return None, None, locals().get("last_error")


def describe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:  # pragma: no cover - introspection edge case
        return f"unavailable: {type(exc).__name__}: {exc}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def resolve_sionna_device(torch_device: Any) -> str | None:
    """Map a torch device to a Sionna-compatible device string."""
    device_str = str(torch_device)
    if device_str == "cuda":
        return "cuda:0"
    return device_str


def format_audit_markdown(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Sionna Native OFDM Component Audit",
        "",
        f"- Sionna import ok: `{payload['sionna_import_ok']}`",
        f"- Sionna version: `{payload['sionna_version']}`",
        f"- Recommended next component to integrate: `{payload['recommended_next_component']}`",
        "",
        "| Component | Import OK | Module | Constructor Signature | Minimal Callable | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["components"]:
        notes = "; ".join(row["notes"]) if row["notes"] else ""
        lines.append(
            f"| {row['name']} | {row['import_ok']} | {row['module_path'] or ''} | "
            f"`{row['constructor_signature'] or ''}` | {row['minimal_callable_example_available']} | {notes} |"
        )
    return lines


def format_baseline_markdown(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Sionna Native OFDM Baseline Chain",
        "",
        f"- Demo status: `{payload['demo_status']}`",
        f"- Sionna import ok: `{payload['sionna_import_ok']}`",
        f"- Sionna version: `{payload['sionna_version']}`",
        f"- Used Sionna native components: `{payload['used_sionna_native_components']}`",
        f"- Fallback used: `{payload['fallback_used']}`",
        "",
        f"- BER if available: `{payload['ber_if_available']}`",
        f"- Symbol MSE: `{payload['symbol_mse']}`",
        f"- Empirical SNR dB: `{payload['empirical_snr_db']}`",
        "",
        "## Components",
    ]
    for component in payload["used_components"]:
        lines.append(f"- `{component}`")
    lines.extend(["", "## Notes"])
    for note in payload["notes"]:
        lines.append(f"- {note}")
    return lines
