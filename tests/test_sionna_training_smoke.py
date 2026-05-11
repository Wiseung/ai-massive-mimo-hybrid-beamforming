from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from beamforming.utils.sionna_env import collect_sionna_env_info


@pytest.mark.skipif(not collect_sionna_env_info()["sionna_import_ok"], reason="Sionna is optional")
def test_sionna_training_smoke_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "sionna_ofdm_smoke_run"
    eval_dir = tmp_path / "sionna_ofdm_smoke_eval"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_sionna_ofdm_beamformer.py",
            "--config",
            "configs/sionna_ofdm_learned_beamformer.yaml",
            "--out",
            str(run_dir),
            "--smoke",
        ],
        check=True,
        cwd=repo_root,
    )
    payload = json.loads((run_dir / "smoke_summary.json").read_text(encoding="utf-8"))
    assert payload["sionna_import_ok"] is True
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "train_log.csv").exists()

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_sionna_ofdm_beamformer.py",
            "--config",
            "configs/sionna_ofdm_learned_beamformer.yaml",
            "--ckpt",
            str(run_dir / "best.pt"),
            "--out",
            str(eval_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    assert (eval_dir / "metrics.csv").exists()
    assert (eval_dir / "summary.md").exists()
