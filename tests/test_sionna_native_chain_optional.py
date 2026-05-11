from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from beamforming.utils.sionna_env import collect_sionna_env_info


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_native_ofdm_component_audit_runs(tmp_path: Path) -> None:
    out_path = tmp_path / "ofdm_component_audit.json"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/audit_sionna_native_ofdm_components.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sionna_import_ok"] is True
    names = {row["name"] for row in payload["components"]}
    assert "ResourceGrid" in names
    assert "OFDMChannel" in names


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_native_ofdm_baseline_chain_runs(tmp_path: Path) -> None:
    out_path = tmp_path / "baseline_chain_summary.json"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/sionna_native_ofdm_baseline_chain.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sionna_import_ok"] is True
    assert payload["demo_status"] == "ok"
    assert "ResourceGrid" in payload["used_components"]
