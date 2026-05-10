from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from beamforming.utils.sionna_env import collect_sionna_env_info, format_sionna_env_lines


def test_sionna_env_check_is_optional() -> None:
    info = collect_sionna_env_info()
    assert "python_version" in info
    assert "torch_version" in info
    assert "cuda_available" in info
    assert "sionna_import_ok" in info
    lines = format_sionna_env_lines(info)
    assert any(line.startswith("Python version:") for line in lines)
    if info["sionna_import_ok"]:
        assert info["sionna_version"] is not None
    else:
        assert "pip install sionna-no-rt" in info["install_hint"]


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_inspect_sionna_api_script_runs(tmp_path: Path) -> None:
    out_path = tmp_path / "sionna_api_summary.json"
    subprocess.run(
        [sys.executable, "scripts/inspect_sionna_api.py", "--out", str(out_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sionna_import_ok"] is True
    assert "sionna.phy" in payload["modules"]


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_phy_awgn_demo_runs(tmp_path: Path) -> None:
    out_path = tmp_path / "sionna_phy_awgn_summary.json"
    subprocess.run(
        [sys.executable, "scripts/sionna_phy_awgn_demo.py", "--out", str(out_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sionna_import_ok"] is True
    assert payload["demo_status"] == "ok"
