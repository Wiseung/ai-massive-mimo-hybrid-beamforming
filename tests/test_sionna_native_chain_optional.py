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


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_precoding_audit_and_beamforming_chain_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    audit_out = tmp_path / "precoding_component_audit.json"
    beamforming_out = tmp_path / "beamforming_chain_summary.json"

    subprocess.run(
        [sys.executable, "scripts/audit_sionna_precoding_components.py", "--out", str(audit_out)],
        check=True,
        cwd=repo_root,
    )
    audit_payload = json.loads(audit_out.read_text(encoding="utf-8"))
    assert audit_payload["sionna_import_ok"] is True
    assert any(row["name"] == "RZFPrecoder" for row in audit_payload["components"])

    subprocess.run(
        [sys.executable, "scripts/sionna_native_ofdm_beamforming_chain.py", "--out", str(beamforming_out)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(beamforming_out.read_text(encoding="utf-8"))
    assert payload["demo_status"] == "ok"
    assert any(row["method"] == "project_rzf" for row in payload["metrics"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_compare_sionna_native_chains_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_out = tmp_path / "baseline_chain_summary.json"
    beamforming_out = tmp_path / "beamforming_chain_summary.json"
    compare_dir = tmp_path / "compare"

    subprocess.run(
        [sys.executable, "scripts/sionna_native_ofdm_baseline_chain.py", "--out", str(baseline_out)],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [sys.executable, "scripts/sionna_native_ofdm_beamforming_chain.py", "--out", str(beamforming_out)],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_sionna_native_chains.py",
            "--baseline",
            str(baseline_out),
            "--beamforming",
            str(beamforming_out),
            "--metrics",
            str(beamforming_out.with_name("beamforming_chain_metrics.csv")),
            "--out",
            str(compare_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (compare_dir / "native_chain_comparison.md").exists()
