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


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_validate_sionna_native_precoder_contract_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_precoder_contract_validation.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_sionna_native_precoder_contract.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "contract_valid" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_demo_sionna_native_precoder_contract_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_precoder_contract_demo.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/demo_sionna_native_precoder_contract.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "relationship_status" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_native_precoder_contract_matrix_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_precoder_contract_matrix.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/test_sionna_native_precoder_contract_matrix.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert payload["aliasing_project_rzf_detected"] is False


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_generate_sionna_interface_rc_artifact_manifest_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "interface_rc_artifact_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_sionna_interface_rc_artifact_manifest.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert any(row["interface_layer"] == "contract_hardening" for row in payload["artifacts"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_reproduce_sionna_interface_rc_minimal_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sionna_interface_rc_minimal_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_sionna_interface_rc_minimal.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "contract_matrix_passed" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_generate_sionna_interface_stable_artifact_manifest_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "interface_stable_artifact_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_sionna_interface_stable_artifact_manifest.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert any(row["release_stage"] == "stable" for row in payload["artifacts"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_reproduce_sionna_interface_stable_minimal_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sionna_interface_stable_minimal_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_sionna_interface_stable_minimal.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "stable_readiness_passed" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_release_body_consistency_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "release_body_consistency.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_release_body_consistency.py",
            "--tag",
            "v1.0.0",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["v1_0_0_title_consistent"] is True


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_artifact_reproducibility_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "artifact_reproducibility_audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_artifact_reproducibility.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["stable_minimal_status_ok"] is True


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_run_optional_sionna_regression_monitor_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "optional_sionna_regression_monitor.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_optional_sionna_regression_monitor.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert any(row["scenario"] == "force_sionna_missing_skip" for row in payload["scenarios"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_release_tag_health_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "release_tag_health.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_release_tag_health.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["latest_release"] == "v1.0.2"


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_generate_release_health_dashboard_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "release_health_dashboard.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_release_health_dashboard.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "overall_status" in payload


def test_run_local_dependency_audit_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "local_dependency_audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_local_dependency_audit.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "pip_check_status" in payload
    assert "pip_audit_available" in payload


def test_generate_security_maintenance_dashboard_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dep_audit = repo_root / "outputs/maintenance/local_dependency_audit.json"
    if not dep_audit.exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/run_local_dependency_audit.py",
                "--out",
                str(dep_audit),
            ],
            check=True,
            cwd=repo_root,
        )
    out_path = tmp_path / "security_maintenance_dashboard.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_security_maintenance_dashboard.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "overall_security_maintenance_status" in payload


def test_run_manual_pip_audit_graceful_warning(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "manual_pip_audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_manual_pip_audit.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["manual_audit_attempted"] is True
    assert "recommended_next_action" in payload
    if payload["pip_audit_available"] is False:
        assert "pip_audit_not_installed" in payload["warnings"]


def test_generate_security_dashboard_consumes_manual_audit(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manual_audit_path = repo_root / "outputs/maintenance/manual_pip_audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_manual_pip_audit.py",
            "--out",
            str(manual_audit_path),
        ],
        check=True,
        cwd=repo_root,
    )
    out_path = tmp_path / "security_maintenance_dashboard.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_security_maintenance_dashboard.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["required_ci_unchanged"] is True
    if payload["warnings"]:
        assert payload["recommended_next_action"] in {
            "run_manual_audit",
            "review_dependency_alerts",
            "install_pip_audit_and_rerun",
        }
