from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from beamforming.utils.sionna_env import collect_sionna_env_info


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_ofdm_benchmark_scripts_quick(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    multiseed_out = tmp_path / "multiseed"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_sionna_ofdm_multiseed_benchmark.py",
            "--configs",
            "configs/sionna_ofdm_learned_beamformer.yaml",
            "configs/sionna_ofdm_residual_rzf.yaml",
            "configs/sionna_ofdm_unfolded_lite.yaml",
            "--seeds",
            "1",
            "--out",
            str(multiseed_out),
            "--quick",
        ],
        check=True,
        cwd=repo_root,
    )
    assert (multiseed_out / "multiseed_summary.csv").exists()

    latency_out = tmp_path / "latency"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_sionna_ofdm_models.py",
            "--out",
            str(latency_out),
            "--warmup-runs",
            "2",
            "--timed-runs",
            "3",
        ],
        check=True,
        cwd=repo_root,
    )
    assert (latency_out / "latency_table.csv").exists()

    scale_out = tmp_path / "scale"
    subprocess.run(
        [
            sys.executable,
            "scripts/sweep_sionna_ofdm_scale.py",
            "--quick",
            "--out",
            str(scale_out),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (scale_out / "scale_sweep.csv").exists()

    snr_out = tmp_path / "snr"
    subprocess.run(
        [
            sys.executable,
            "scripts/sweep_sionna_train_snr.py",
            "--model",
            "sionna_ofdm_residual_rzf",
            "--quick",
            "--out",
            str(snr_out),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (snr_out / "snr_ablation.csv").exists()
