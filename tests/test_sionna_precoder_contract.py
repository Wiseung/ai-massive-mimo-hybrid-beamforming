from __future__ import annotations

import pytest

from beamforming.utils.sionna_precoder_api_bridge import build_sionna_rzf_precoder_contract


def test_sionna_precoder_contract_required_fields_present() -> None:
    contract = build_sionna_rzf_precoder_contract(
        sionna_version="2.0.1",
        callable=True,
        converted_precoder_output_shape=[16, 16, 16, 4],
        relationship_to_project_rzf="close_but_different",
        strict_equivalence_claim_allowed=False,
        semantic_compatibility_passed=True,
        project_side_precoder=False,
        sionna_native_precoder=True,
        full_native_only=False,
    )
    summary = contract.summary_dict()
    assert summary["method_name"] == "sionna_rzf_precoder"
    assert summary["sionna_component"] == "sionna.phy.ofdm.RZFPrecoder"
    assert summary["expected_x_shape"] == ["B", "num_tx", "num_streams_per_tx", "num_ofdm_symbols", "fft_size"]
    assert summary["expected_h_shape"] == ["B", "num_rx", "num_rx_ant", "num_tx", "num_tx_ant", "num_ofdm_symbols", "fft_size"]
    assert summary["expected_x_precoded_shape"] == ["B", "num_tx", "num_tx_ant", "num_ofdm_symbols", "fft_size"]


def test_sionna_precoder_contract_strict_equivalence_false() -> None:
    contract = build_sionna_rzf_precoder_contract(
        sionna_version="2.0.1",
        callable=True,
        converted_precoder_output_shape=[16, 16, 16, 4],
        relationship_to_project_rzf="close_but_different",
        strict_equivalence_claim_allowed=False,
        semantic_compatibility_passed=True,
        project_side_precoder=False,
        sionna_native_precoder=True,
        full_native_only=False,
    )
    assert contract.strict_equivalence_claim_allowed is False
    assert contract.full_native_only is False


def test_sionna_native_precoder_true_only_for_native_adapter_output() -> None:
    contract = build_sionna_rzf_precoder_contract(
        sionna_version="2.0.1",
        callable=True,
        converted_precoder_output_shape=[16, 16, 16, 4],
        relationship_to_project_rzf="close_but_different",
        strict_equivalence_claim_allowed=False,
        semantic_compatibility_passed=True,
        project_side_precoder=False,
        sionna_native_precoder=True,
        full_native_only=False,
    )
    assert contract.sionna_native_precoder is True
    assert contract.project_side_precoder is False


def test_project_rzf_strict_equivalence_not_claimed() -> None:
    contract = build_sionna_rzf_precoder_contract(
        sionna_version="2.0.1",
        callable=True,
        converted_precoder_output_shape=[16, 16, 16, 4],
        relationship_to_project_rzf="close_but_different",
        strict_equivalence_claim_allowed=False,
        semantic_compatibility_passed=True,
        project_side_precoder=False,
        sionna_native_precoder=True,
        full_native_only=False,
    )
    assert contract.relationship_to_project_rzf == "close_but_different"
    assert contract.strict_equivalence_claim_allowed is False


def test_incomplete_contract_raises_clear_error() -> None:
    contract = build_sionna_rzf_precoder_contract(
        sionna_version="2.0.1",
        callable=True,
        converted_precoder_output_shape=[16, 16, 16, 4],
        relationship_to_project_rzf="close_but_different",
        strict_equivalence_claim_allowed=False,
        semantic_compatibility_passed=True,
        project_side_precoder=False,
        sionna_native_precoder=True,
        full_native_only=False,
    )
    contract.expected_h_shape = ["B", "num_rx"]
    with pytest.raises(ValueError, match="expected_h_shape_must_have_len_7"):
        contract.validate_contract()
