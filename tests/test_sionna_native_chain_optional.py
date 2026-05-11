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


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_pilot_audit_and_estimator_demo_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pilot_out = tmp_path / "pilot_pattern_audit.json"
    demo_out = tmp_path / "estimator_equalizer_demo_summary.json"
    subprocess.run(
        [sys.executable, "scripts/audit_sionna_resource_grid_pilots.py", "--out", str(pilot_out)],
        check=True,
        cwd=repo_root,
    )
    pilot_payload = json.loads(pilot_out.read_text(encoding="utf-8"))
    assert pilot_payload["summary"]["pilot_indices_required"] is True
    subprocess.run(
        [sys.executable, "scripts/sionna_native_estimator_equalizer_demo.py", "--out", str(demo_out)],
        check=True,
        cwd=repo_root,
    )
    demo_payload = json.loads(demo_out.read_text(encoding="utf-8"))
    assert demo_payload["used_sionna_estimator"] is True
    assert demo_payload["used_sionna_equalizer"] is True


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_beamforming_receiver_chain_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_out = tmp_path / "baseline_chain_summary.json"
    beamforming_out = tmp_path / "beamforming_chain_summary.json"
    receiver_out = tmp_path / "beamforming_receiver_chain_v2_summary.json"
    compare_dir = tmp_path / "compare_v2"
    shape_trace_out = tmp_path / "beamformed_receiver_shape_trace.json"
    stream_mgmt_out = tmp_path / "stream_management_audit.json"
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
        [sys.executable, "scripts/trace_sionna_beamformed_receiver_shapes.py", "--out", str(shape_trace_out)],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [sys.executable, "scripts/audit_sionna_stream_management.py", "--out", str(stream_mgmt_out)],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/sionna_native_ofdm_beamforming_chain.py",
            "--out",
            str(receiver_out),
            "--enable-receiver-chain",
            "--receiver-mode",
            "auto",
            "--trace-shapes",
        ],
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
            "--receiver",
            str(receiver_out),
            "--metrics",
            str(receiver_out.with_name("beamforming_receiver_chain_v2_metrics.csv")),
            "--out",
            str(compare_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    shape_payload = json.loads(shape_trace_out.read_text(encoding="utf-8"))
    assert "beamformed_paths" in shape_payload
    stream_payload = json.loads(stream_mgmt_out.read_text(encoding="utf-8"))
    assert stream_payload["summary"]["recommended_configuration"]["num_streams_per_tx"] == 4
    receiver_payload = json.loads(receiver_out.read_text(encoding="utf-8"))
    assert receiver_payload["native_receiver_attempted"] is True
    if not receiver_payload["native_receiver_success"]:
        assert receiver_payload["native_failure_stage"]
        assert receiver_payload["native_failure_reason"]
    assert (compare_dir / "native_chain_comparison_v2.md").exists()


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_native_learned_chain_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    learned_out = tmp_path / "learned_beamforming_receiver_summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/sionna_native_ofdm_learned_beamforming_chain.py",
            "--out",
            str(learned_out),
            "--receiver-mode",
            "auto",
            "--trace-shapes",
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads(learned_out.read_text(encoding="utf-8"))
    assert payload["demo_status"] == "ok"
    assert payload["native_receiver_attempted"] is True
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_sionna_native_learned_beamforming.py",
            "--analytic-summary",
            str(repo_root / "outputs/sionna_native_chain/beamforming_receiver_chain_v2_summary.json"),
            "--analytic-metrics",
            str(repo_root / "outputs/sionna_native_chain/beamforming_receiver_chain_v2_metrics.csv"),
            "--learned-summary",
            str(learned_out),
            "--learned-metrics",
            str(learned_out.with_name("learned_beamforming_receiver_metrics.csv")),
            "--out",
            str(tmp_path / "learned_compare"),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (tmp_path / "learned_compare" / "native_learned_comparison.md").exists()


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_native_chain_manifest_and_minimal_repro_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_out = tmp_path / "native_chain_artifact_manifest.json"
    repro_out = tmp_path / "sionna_native_chain_minimal_summary.json"
    subprocess.run(
        [sys.executable, "scripts/generate_sionna_native_chain_artifact_manifest.py", "--out", str(manifest_out)],
        check=True,
        cwd=repo_root,
    )
    manifest_payload = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest_payload["artifacts"]
    subprocess.run(
        [sys.executable, "scripts/reproduce_sionna_native_chain_minimal.py", "--out", str(repro_out)],
        check=True,
        cwd=repo_root,
    )
    repro_payload = json.loads(repro_out.read_text(encoding="utf-8"))
    assert repro_payload["status"] == "ok"
    assert repro_payload["baseline_receiver_check"]["used_sionna_channel"] is True
