from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from beamforming.utils.sionna_env import collect_sionna_env_info


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_channel_tensor_audit_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "channel_tensor_audit.json"
    subprocess.run(
        [sys.executable, "scripts/audit_sionna_channel_tensor_shapes.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "components" in payload
    assert payload["summary"]["ofdmchannel_returns_channel_tensor"] in {True, False}


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_extract_channel_hf_demo_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "extract_h_f_demo_summary.json"
    subprocess.run(
        [sys.executable, "scripts/sionna_extract_channel_hf_demo.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "extraction_success" in payload
    assert "project_h_f_shape_compatible" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_native_channel_beamforming_chain_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_channel_beamforming_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/sionna_native_channel_beamforming_chain.py",
            "--out",
            str(out_path),
            "--receiver-mode",
            "auto",
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "project_h_f_assisted" in payload
