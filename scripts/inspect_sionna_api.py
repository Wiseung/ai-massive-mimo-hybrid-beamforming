#!/usr/bin/env python
"""Inspect the currently installed Sionna 2.x API surface."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info


MODULE_NAMES = [
    "sionna.phy",
    "sionna.phy.channel",
    "sionna.phy.ofdm",
    "sionna.phy.mapping",
    "sionna.phy.fec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def inspect_module(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        public_names = sorted(item for item in dir(module) if not item.startswith("_"))
        return {
            "import_ok": True,
            "public_name_count": len(public_names),
            "public_names_preview": public_names[:30],
        }
    except Exception as exc:  # pragma: no cover - script-path behavior
        return {
            "import_ok": False,
            "public_name_count": 0,
            "public_names_preview": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_info = collect_sionna_env_info()
    payload: dict[str, object] = {
        "python_version": env_info["python_version"],
        "torch_version": env_info["torch_version"],
        "sionna_import_ok": env_info["sionna_import_ok"],
        "sionna_version": env_info["sionna_version"],
        "modules": {},
    }

    modules: dict[str, object] = {}
    if env_info["sionna_import_ok"]:
        for name in MODULE_NAMES:
            modules[name] = inspect_module(name)
    else:
        for name in MODULE_NAMES:
            modules[name] = {
                "import_ok": False,
                "public_name_count": 0,
                "public_names_preview": [],
                "error": env_info["sionna_error"],
            }
        payload["install_hint"] = env_info["install_hint"]
    payload["modules"] = modules

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Sionna API summary to {out_path}")


if __name__ == "__main__":
    main()
