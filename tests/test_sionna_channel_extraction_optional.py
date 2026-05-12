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
            "--seed",
            "0",
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "project_h_f_assisted" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_validate_sionna_extracted_hf_axes_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "hf_axis_validation.json"
    subprocess.run(
        [sys.executable, "scripts/validate_sionna_extracted_hf_axes.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["validation_status"] in {"ok", "skipped"}
    assert "axis_spot_check_passed" in payload
    assert payload["csi_interface_used"] is True


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_sionna_csi_interface_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "csi_interface_audit.json"
    subprocess.run(
        [sys.executable, "scripts/audit_sionna_csi_interface.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped", "failed"}
    assert "csi_interface_used" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_csi_backed_beamforming_chain_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "csi_backed_beamforming_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/sionna_csi_backed_beamforming_chain.py",
            "--out",
            str(out_path),
            "--receiver-mode",
            "auto",
            "--seed",
            "0",
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["csi_interface_used"] is True
    assert "metrics" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_benchmark_sionna_extracted_h_consistency_quick_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "extracted_h_consistency"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_sionna_extracted_h_consistency.py",
            "--out",
            str(out_dir),
            "--seeds",
            "1",
            "--snrs",
            "0",
            "10",
            "--quick",
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "summary.md").exists()


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sweep_sionna_channel_extraction_config_quick_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "extraction_config_sweep"
    subprocess.run(
        [
            sys.executable,
            "scripts/sweep_sionna_channel_extraction_config.py",
            "--quick",
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "extraction_sweep.csv").exists()
    assert (out_dir / "extraction_sweep.md").exists()


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_compare_project_hf_vs_extracted_hf_runs_if_inputs_exist(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    project = repo_root / "outputs/sionna_native_chain/learned_beamforming_receiver_metrics.csv"
    extracted = repo_root / "outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv"
    consistency = repo_root / "outputs/sionna_channel_extraction/extracted_h_consistency/metrics.csv"
    if not (project.exists() and extracted.exists() and consistency.exists()):
        pytest.skip("Required comparison artifacts not present")
    out_dir = tmp_path / "project_vs_extracted_hf"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_project_hf_vs_extracted_hf.py",
            "--project",
            str(project),
            "--extracted",
            str(extracted),
            "--consistency",
            str(consistency),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "comparison.csv").exists()
    assert (out_dir / "comparison.md").exists()


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_compare_csi_backed_vs_raw_extracted_h_runs_if_inputs_exist(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw = repo_root / "outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv"
    csi = repo_root / "outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv"
    if not (raw.exists() and csi.exists()):
        pytest.skip("Required raw/csi comparison artifacts not present")
    out_dir = tmp_path / "csi_compare"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_csi_backed_vs_raw_extracted_h.py",
            "--raw",
            str(raw),
            "--csi",
            str(csi),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "csi_interface_comparison.csv").exists()
    assert (out_dir / "csi_interface_comparison.md").exists()
    payload = (out_dir / "csi_interface_comparison.md").read_text(encoding="utf-8")
    assert "comparison_type: `cross_run_comparison`" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_validate_csi_same_batch_equivalence_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "csi_same_batch_equivalence.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_csi_same_batch_equivalence.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "same_channel_tensor_used" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_csi_raw_comparison_mismatch_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw = repo_root / "outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv"
    csi = repo_root / "outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv"
    if not (raw.exists() and csi.exists()):
        pytest.skip("Required artifacts not present")
    out_dir = tmp_path / "mismatch_audit"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_csi_raw_comparison_mismatch.py",
            "--raw",
            str(raw),
            "--csi",
            str(csi),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads((out_dir / "csi_raw_mismatch_audit.json").read_text(encoding="utf-8"))
    assert payload["comparison_independent_runs"] is True


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_csi_consumers_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "csi_consumer_audit.json"
    subprocess.run(
        [sys.executable, "scripts/audit_csi_consumers.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["total_consumers_audited"] >= 1
    assert "priority_migration_targets" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_demo_unified_csi_consumers_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "unified_csi_consumers_summary.json"
    subprocess.run(
        [sys.executable, "scripts/demo_unified_csi_consumers.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped", "failed"}
    assert "all_consumers_accept_csi" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_precoder_interface_consumers_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "precoder_interface_audit.json"
    subprocess.run(
        [sys.executable, "scripts/audit_precoder_interface_consumers.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["total_consumers_audited"] >= 1
    assert "high_priority_raw_only_gap" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_demo_unified_csi_and_precoder_interfaces_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "unified_csi_precoder_summary.json"
    subprocess.run(
        [sys.executable, "scripts/demo_unified_csi_and_precoder_interfaces.py", "--out", str(out_path)],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped", "failed"}
    assert "all_precoders_emit_precoder_output" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_compare_raw_ff_vs_precoder_output_runs_if_inputs_exist(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw = repo_root / "outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv"
    precoder_output = repo_root / "outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv"
    if not (raw.exists() and precoder_output.exists()):
        pytest.skip("Required raw/precoder-output artifacts not present")
    out_dir = tmp_path / "precoder_compare"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_raw_ff_vs_precoder_output.py",
            "--raw",
            str(raw),
            "--precoder-output",
            str(precoder_output),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "precoder_output_comparison.csv").exists()
    payload = (out_dir / "precoder_output_comparison.md").read_text(encoding="utf-8")
    assert "comparison_type: `cross_run_comparison`" in payload
    assert "same_batch_comparison: `false`" in payload
    assert "interface_bug_evidence: `false`" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_validate_precoder_output_same_batch_equivalence_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "precoder_output_same_batch_equivalence.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_precoder_output_same_batch_equivalence.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "same_csi_object_used" in payload
    assert "strict_equivalence_claim_allowed" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_precoder_output_comparison_mismatch_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw = repo_root / "outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv"
    precoder_output = repo_root / "outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv"
    if not (raw.exists() and precoder_output.exists()):
        pytest.skip("Required raw/precoder-output artifacts not present")
    out_dir = tmp_path / "precoder_mismatch_audit"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_precoder_output_comparison_mismatch.py",
            "--raw",
            str(raw),
            "--precoder-output",
            str(precoder_output),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads((out_dir / "precoder_output_mismatch_audit.json").read_text(encoding="utf-8"))
    assert payload["comparison_independent_runs"] is True
    assert payload["comparison_type"] == "cross_run_comparison"


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_generate_sionna_precoder_interface_artifact_manifest_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "precoder_interface_artifact_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_sionna_precoder_interface_artifact_manifest.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert any(row["name"] == "precoder_output_same_batch_equivalence" for row in payload["artifacts"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_reproduce_sionna_precoder_interface_minimal_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sionna_precoder_interface_minimal_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_sionna_precoder_interface_minimal.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "precoder_interface_used" in payload or payload["status"] == "skipped"


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_audit_sionna_native_precoder_api_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_precoder_api_audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_sionna_native_precoder_api.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sionna_import_ok"] is True
    assert "summary" in payload
    assert "recommended_next_step" in payload["summary"]


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_probe_sionna_rzf_precoder_bridge_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "rzf_precoder_probe_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/probe_sionna_rzf_precoder_bridge.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "failed", "skipped"}
    assert "sionna_rzf_callable" in payload


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_compare_project_vs_sionna_precoder_runs_if_inputs_exist(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    probe = repo_root / "outputs/sionna_precoder_api/rzf_precoder_probe_summary.json"
    metrics = repo_root / "outputs/sionna_precoder_api/rzf_precoder_probe_metrics.csv"
    if not (probe.exists() and metrics.exists()):
        pytest.skip("Required Sionna precoder probe artifacts not present")
    out_dir = tmp_path / "project_vs_sionna_precoder"
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


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_compare_unified_csi_consumers_runs_if_inputs_exist(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline = repo_root / "outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv"
    unified = repo_root / "outputs/sionna_channel_extraction/unified_csi_consumers_metrics.csv"
    if not (baseline.exists() and unified.exists()):
        pytest.skip("Required baseline/unified artifacts not present")
    out_dir = tmp_path / "unified_compare"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_unified_csi_consumers.py",
            "--baseline",
            str(baseline),
            "--unified",
            str(unified),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (out_dir / "unified_csi_consumer_comparison.csv").exists()
    assert (out_dir / "unified_csi_consumer_comparison.md").exists()


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_generate_sionna_csi_interface_artifact_manifest_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "csi_interface_artifact_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_sionna_csi_interface_artifact_manifest.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert any(row["name"] == "csi_same_batch_equivalence" for row in payload["artifacts"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_reproduce_sionna_csi_interface_minimal_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sionna_csi_interface_minimal_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_sionna_csi_interface_minimal.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "same_batch_equivalence_passed" in payload or payload["status"] == "skipped"


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_generate_sionna_csi_consumer_artifact_manifest_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "csi_consumer_artifact_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_sionna_csi_consumer_artifact_manifest.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert any(row["name"] == "unified_csi_consumers_summary" for row in payload["artifacts"])


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_reproduce_sionna_csi_consumer_minimal_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sionna_csi_consumer_minimal_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_sionna_csi_consumer_minimal.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "all_consumers_accept_csi" in payload or payload["status"] == "skipped"


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_generate_sionna_channel_extraction_artifact_manifest_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "channel_extraction_artifact_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_sionna_channel_extraction_artifact_manifest.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert isinstance(payload["artifacts"], list)


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_reproduce_sionna_channel_extraction_minimal_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sionna_channel_extraction_minimal_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_sionna_channel_extraction_minimal.py",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"ok", "skipped"}
    assert "full_native_only" in payload or payload["status"] == "skipped"
