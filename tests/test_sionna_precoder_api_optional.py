from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from beamforming.utils.sionna_env import collect_sionna_env_info


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_sionna_native_precoder_api_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_precoder_api_audit.json"
    subprocess.run(
        [sys.executable, "scripts/audit_sionna_native_precoder_api.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sionna_import_ok"] is True
    assert payload["summary"]["sionna_rzf_precoder_available"] in {True, False}
    assert any(row["name"] == "RZFPrecoder" for row in payload["targets"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_probe_sionna_rzf_precoder_bridge_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "rzf_precoder_probe_summary.json"
    subprocess.run(
        [sys.executable, "scripts/probe_sionna_rzf_precoder_bridge.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "failed", "skipped"}
    assert "sionna_rzf_callable" in payload
    assert "converted_to_precoder_output" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_compare_project_vs_sionna_precoder_runs_if_probe_exists(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    probe = repo_root / "outputs/sionna_precoder_api/rzf_precoder_probe_summary.json"
    metrics = repo_root / "outputs/sionna_precoder_api/rzf_precoder_probe_metrics.csv"
    if not (probe.exists() and metrics.exists()):
        pytest.skip("Required Sionna precoder probe artifacts not present")
    out_dir = tmp_path / "sionna_precoder_compare"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_project_vs_sionna_precoder.py",
            "--probe",
            str(probe),
            "--metrics",
            str(metrics),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "project_vs_sionna_precoder_comparison.csv").exists()
    md_text = (out_dir / "project_vs_sionna_precoder_comparison.md").read_text(encoding="utf-8")
    assert "full native-only benchmark: `False`" in md_text
