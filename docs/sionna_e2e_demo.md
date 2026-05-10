# Sionna E2E Demo Branch

## Goal

- feature branch only
- optional Sionna smoke validation
- no change to the `v0.1.0` benchmark claims

## Environment

```bash
conda create -n mimo-sionna python=3.11
conda activate mimo-sionna
pip install torch torchvision torchaudio
pip install sionna-no-rt
pip install -e ".[dev]"
```

Recommended practice:

- use a dedicated conda environment
- start with `sionna-no-rt`
- keep this branch separate from the released benchmark path

## Commands

```bash
python scripts/check_sionna_env.py
python scripts/inspect_sionna_api.py --out outputs/sionna_smoke/sionna_api_summary.json
python scripts/sionna_smoke_demo.py --out outputs/sionna_smoke/sionna_smoke_summary.json
python scripts/sionna_bridge_beamforming_demo.py --out outputs/sionna_smoke/bridge_beamforming_summary.json
python scripts/sionna_phy_awgn_demo.py --out outputs/sionna_smoke/sionna_phy_awgn_summary.json
python scripts/sionna_phy_beamforming_link_demo.py --out outputs/sionna_smoke/sionna_phy_beamforming_link_summary.json
```

## Current Status

- the optional Sionna smoke demo is complete
- the API introspection step confirms `sionna.phy`, `sionna.phy.channel`, `sionna.phy.ofdm`, `sionna.phy.mapping`, and `sionna.phy.fec` availability on this branch environment
- the PHY AWGN demo uses Sionna PHY when the inspected `AWGN` component is callable; otherwise it falls back to a torch AWGN path and records that explicitly
- the beamforming link demo likewise prefers Sionna PHY AWGN and otherwise records explicit fallback

## Current Limitations

- not a full Sionna end-to-end link yet
- no Sionna RT yet
- no ray tracing yet
- no 5G NR full-stack yet
- does not change the `v0.1.0` benchmark claims

## Future Work

- Sionna PHY OFDM link
- replace torch fallback with confirmed Sionna PHY components where needed
- differentiable beamforming module
- optional Sionna RT channel generation
- compare with DeepMIMO channels
