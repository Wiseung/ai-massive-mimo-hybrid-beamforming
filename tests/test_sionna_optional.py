from __future__ import annotations

from beamforming.utils.sionna_env import collect_sionna_env_info, format_sionna_env_lines


def test_sionna_env_check_is_optional() -> None:
    info = collect_sionna_env_info()
    assert "python_version" in info
    assert "torch_version" in info
    assert "cuda_available" in info
    assert "sionna_import_ok" in info
    lines = format_sionna_env_lines(info)
    assert any(line.startswith("Python version:") for line in lines)
    if info["sionna_import_ok"]:
        assert info["sionna_version"] is not None
    else:
        assert "pip install sionna-no-rt" in info["install_hint"]
