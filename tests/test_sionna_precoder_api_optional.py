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
def test_validate_sionna_rzf_same_realization_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sionna_rzf_same_realization.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_sionna_rzf_same_realization.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert payload["comparison_type"] == "same_realization_comparison"
    assert "strict_equivalence_claim_allowed" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_benchmark_sionna_rzf_precoder_alignment_quick_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "sionna_rzf_alignment_quick"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_sionna_rzf_precoder_alignment.py",
            "--quick",
            "--seeds",
            "1",
            "2",
            "3",
            "--snrs",
            "0",
            "5",
            "10",
            "15",
            "20",
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "summary.md").exists()


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_compare_project_vs_sionna_precoder_runs_if_inputs_exist(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    same_realization = repo_root / "outputs/sionna_precoder_api/sionna_rzf_same_realization.json"
    alignment = repo_root / "outputs/sionna_precoder_api/sionna_rzf_alignment_quick/metrics.csv"
    unified = repo_root / "outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv"
    if not (same_realization.exists() and alignment.exists() and unified.exists()):
        pytest.skip("Required Sionna precoder alignment artifacts not present")
    out_dir = tmp_path / "sionna_precoder_compare"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_project_vs_sionna_precoder.py",
            "--same-realization",
            str(same_realization),
            "--alignment",
            str(alignment),
            "--unified",
            str(unified),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "project_vs_sionna_precoder_comparison_v2.csv").exists()
    md_text = (out_dir / "project_vs_sionna_precoder_comparison_v2.md").read_text(encoding="utf-8")
    assert "full native-only benchmark: `False`" in md_text


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_generate_sionna_native_precoder_artifact_manifest_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_precoder_artifact_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_sionna_native_precoder_artifact_manifest.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert any(row["name"] == "sionna_rzf_same_realization" for row in payload["artifacts"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_reproduce_sionna_native_precoder_minimal_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sionna_native_precoder_minimal_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_sionna_native_precoder_minimal.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "sionna_rzf_callable" in payload
