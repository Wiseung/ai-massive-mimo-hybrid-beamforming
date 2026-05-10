# Sionna Artifact Manifest

- generated_from_commit: `4fa413bff5625f0996a238e6092c8c17bb646baf`
- note: optional Sionna smoke artifacts only; full end-to-end, RT, and 5G NR full-stack claims remain out of scope.

| path | exists | real_sionna | fallback_used | full_e2e | command |
| --- | --- | --- | --- | --- | --- |
| outputs/sionna_smoke/sionna_api_summary.json | True | False | None | False | `python scripts/inspect_sionna_api.py --out outputs/sionna_smoke/sionna_api_summary.json` |
| outputs/sionna_smoke/sionna_phy_awgn_summary.json | True | True | False | False | `python scripts/sionna_phy_awgn_demo.py --out outputs/sionna_smoke/sionna_phy_awgn_summary.json` |
| outputs/sionna_smoke/sionna_phy_beamforming_link_summary.json | True | True | False | False | `python scripts/sionna_phy_beamforming_link_demo.py --out outputs/sionna_smoke/sionna_phy_beamforming_link_summary.json` |
| outputs/sionna_smoke/sionna_ofdm_api_summary.json | True | False | None | False | `python scripts/inspect_sionna_ofdm_api.py --out outputs/sionna_smoke/sionna_ofdm_api_summary.json` |
| outputs/sionna_smoke/sionna_ofdm_resource_grid_summary.json | True | True | False | False | `python scripts/sionna_ofdm_resource_grid_demo.py --out outputs/sionna_smoke/sionna_ofdm_resource_grid_summary.json` |
| outputs/sionna_smoke/sionna_ofdm_beamforming_bridge_summary.json | True | True | False | False | `python scripts/sionna_ofdm_beamforming_bridge_demo.py --out outputs/sionna_smoke/sionna_ofdm_beamforming_bridge_summary.json` |
| outputs/sionna_smoke/differentiable_beamformer_gradcheck.json | True | False | None | False | `python scripts/check_differentiable_beamformer_gradients.py --out outputs/sionna_smoke/differentiable_beamformer_gradcheck.json` |
| outputs/sionna_smoke/sionna_ofdm_differentiable_beamforming_summary.json | True | True | False | False | `python scripts/sionna_ofdm_differentiable_beamforming_demo.py --out outputs/sionna_smoke/sionna_ofdm_differentiable_beamforming_summary.json` |
