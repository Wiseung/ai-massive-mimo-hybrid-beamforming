from __future__ import annotations

import math

import pandas as pd

from beamforming.evaluation import add_relative_gaps, gap_to_reference


def test_gap_formula_matches_project_definition() -> None:
    method = pd.Series([4.0, 6.0])
    reference = pd.Series([5.0, 8.0])
    gap = gap_to_reference(method, reference)
    assert math.isclose(float(gap.iloc[0]), -0.2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(gap.iloc[1]), -0.25, rel_tol=0.0, abs_tol=1e-12)


def test_add_relative_gaps_uses_same_formula() -> None:
    df = pd.DataFrame(
        [
            {"method": "rzf", "snr_db": 10.0, "se": 10.0, "runtime_sec": 0.1},
            {"method": "wmmse", "snr_db": 10.0, "se": 12.0, "runtime_sec": 0.2},
            {"method": "cnn", "snr_db": 10.0, "se": 9.0, "runtime_sec": 0.05},
            {"method": "rzf", "snr_db": 20.0, "se": 20.0, "runtime_sec": 0.1},
            {"method": "wmmse", "snr_db": 20.0, "se": 21.0, "runtime_sec": 0.2},
            {"method": "cnn", "snr_db": 20.0, "se": 18.0, "runtime_sec": 0.05},
        ]
    )
    out = add_relative_gaps(df)
    cnn_10 = out[(out["method"] == "cnn") & (out["snr_db"] == 10.0)].iloc[0]
    cnn_20 = out[(out["method"] == "cnn") & (out["snr_db"] == 20.0)].iloc[0]
    assert math.isclose(float(cnn_10["relative_gap_to_rzf"]), -0.1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(cnn_20["relative_gap_to_rzf"]), -0.1, rel_tol=0.0, abs_tol=1e-12)
    assert cnn_10["gap_formula"] == "(method_se - reference_se) / reference_se"
    assert cnn_10["reference_method"] == "rzf"
