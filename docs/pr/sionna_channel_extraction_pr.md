# PR Title Suggestion

`feat: add Sionna-native channel extraction bridge`

## Summary

This PR adds an optional Sionna-native channel-extraction bridge on top of the existing synthetic OFDM learned-beamforming workflow. The main goal is to reduce the earlier `project-H_f-assisted` limitation by extracting a project-compatible `H_f=(B,Nsc,K,Nt)` from a real Sionna `OFDMChannel(return_channel=True)` tensor, while preserving the already validated native Sionna receiver path.

## What Is Included

- Sionna channel tensor audit
- project-compatible `H_f` extraction utility
- extracted-H axis validation
- native-channel-assisted beamforming chain
- extracted-H consistency quick benchmark
- extraction-config sweep
- project-H_f vs extracted-H_f comparison
- artifact manifest
- minimal reproduction command

## What Is Explicitly Not Included

- Sionna RT
- ray tracing
- 5G NR full stack
- full native-only benchmark
- production e2e
- stable learned `> WMMSE-iter5` claim

## Key Results

- `OFDMChannel(return_channel=True)` returns a usable channel tensor in the current environment.
- The current observed tensor layout is consistent with `h=[B,rx,rx_ant,tx,tx_ant,ofdm_symbol,fft_bin]`.
- The current bridge extracts `H_f=[B,Nsc,K,Nt]`.
- Axis validation passes with `spot_check_max_abs_diff=0.0`.
- The native-channel-assisted beamforming path succeeds with `extraction_success=true` and `native_receiver_success=true`.
- The extracted-H path reduces the project-assisted limitation, but it still does not justify a full native-only claim.

## Quick Benchmark Caveat

The extracted-H consistency benchmark is intentionally a quick/limited validation:

- seeds: `1,2,3`
- SNR: `0,5,10,15,20 dB`
- quick batch size

Current exact proxy/native rank agreement is low, so project proxy metrics should not be used as a substitute for native receiver metrics.

## Not Full Native-Only Caveat

The current supported description is:

- native-channel-assisted
- native-receiver-assisted

It is not a full native-only benchmark because the precoder and learned-model interface remain project-side bridge components.

## Test Results

- `python -m compileall src scripts tests`
- `pytest -q`
- current branch should remain green after this phase

## Reproduction Commands

```bash
python scripts/generate_sionna_channel_extraction_artifact_manifest.py \
  --out outputs/sionna_channel_extraction/channel_extraction_artifact_manifest.json

python scripts/reproduce_sionna_channel_extraction_minimal.py \
  --out outputs/repro/sionna_channel_extraction_minimal_summary.json
```

## Risk Assessment

- moderate integration risk due to optional Sionna APIs and shape assumptions
- low scope risk because RT / ray tracing / 5G NR full stack remain explicitly excluded
- moderate interpretation risk if the extracted-H path is overstated as full native-only

## Merge Recommendation

Merge if the project wants a cleaner optional Sionna channel-extraction bridge with explicit validation artifacts and reviewer-friendly reproduction. Keep the wording conservative: extracted-H reduces project assistance, but does not complete a full native-only benchmark.
