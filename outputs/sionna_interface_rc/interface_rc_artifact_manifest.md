# Interface-first Sionna Bridge RC Artifact Manifest

- generated_from_commit: `916f51bdf265423908019651578cb66403497ea5`
- note: interface-first release candidate only; optional Sionna dependency remains in place.

| name | layer | exists | strict_equivalence_claim_allowed | relationship_status | contract_valid | regression_matrix_passed | command |
| --- | --- | --- | --- | --- | --- | --- | --- |
| channel_extraction_artifact_manifest | channel_extraction | True | None | None | None | None | `python scripts/generate_sionna_channel_extraction_artifact_manifest.py --out outputs/sionna_channel_extraction/channel_extraction_artifact_manifest.json` |
| csi_interface_artifact_manifest | csi_interface | True | None | None | None | None | `python scripts/generate_sionna_csi_interface_artifact_manifest.py --out outputs/sionna_channel_extraction/csi_interface_artifact_manifest.json` |
| csi_consumer_artifact_manifest | csi_consumer | True | None | None | None | None | `python scripts/generate_sionna_csi_consumer_artifact_manifest.py --out outputs/sionna_channel_extraction/csi_consumer_artifact_manifest.json` |
| precoder_output_artifact_manifest | precoder_output | True | None | None | None | None | `python scripts/generate_sionna_precoder_interface_artifact_manifest.py --out outputs/sionna_channel_extraction/precoder_interface_artifact_manifest.json` |
| native_precoder_artifact_manifest | sionna_native_precoder | True | False | close_but_different | None | None | `python scripts/generate_sionna_native_precoder_artifact_manifest.py --out outputs/sionna_precoder_api/native_precoder_artifact_manifest.json` |
| native_precoder_contract_validation | contract_hardening | True | False | close_but_different | True | None | `python scripts/validate_sionna_native_precoder_contract.py --out outputs/sionna_precoder_api/native_precoder_contract_validation.json` |
| native_precoder_contract_demo | contract_hardening | True | False | close_but_different | True | None | `python scripts/demo_sionna_native_precoder_contract.py --out outputs/sionna_precoder_api/native_precoder_contract_demo.json` |
| native_precoder_contract_matrix | contract_hardening | True | None | None | None | True | `python scripts/test_sionna_native_precoder_contract_matrix.py --out outputs/sionna_precoder_api/native_precoder_contract_matrix.json` |
| native_precoder_minimal_reproduction | sionna_native_precoder | True | False | close_but_different | None | None | `python scripts/reproduce_sionna_native_precoder_minimal.py --out outputs/repro/sionna_native_precoder_minimal_summary.json` |
| interface_rc_minimal_reproduction | contract_hardening | True | False | close_but_different | True | True | `python scripts/reproduce_sionna_interface_rc_minimal.py --out outputs/repro/sionna_interface_rc_minimal_summary.json` |
