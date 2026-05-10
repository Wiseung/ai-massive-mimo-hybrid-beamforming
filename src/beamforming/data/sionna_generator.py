"""Sionna channel generation hooks with graceful fallback."""

from __future__ import annotations

from typing import Any


def generate_sionna_demo_channel(*args, **kwargs) -> Any:
    """Placeholder for a minimal Sionna-generated channel demo."""
    try:
        import sionna  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Sionna is not installed. Install the optional dependency before running the Sionna end-to-end demo."
        ) from exc
    raise NotImplementedError(
        "Sionna end-to-end demo scaffold exists, but runtime validation is deferred until Sionna is installed locally."
    )
