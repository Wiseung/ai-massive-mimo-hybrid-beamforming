# Sionna ResourceGrid Pilot Audit

- Current pilot-pattern failure cause: beamforming_chain used pilot_pattern=None, which creates EmptyPilotPattern and causes LSChannelEstimator to raise AssertionError.
- Working LS estimator config: `pilot_pattern='kronecker' with pilot_ofdm_symbol_indices=[0]`
- Need pilot_ofdm_symbol_indices: `True`

| Config | Pilot Pattern | Pilot Symbols | Data Symbols | LS OK | LMMSE ctor OK |
| --- | --- | ---: | ---: | --- | --- |
| empty_default | EmptyPilotPattern | 0 | 52 | False | True |
| kronecker_one_pilot_symbol | KroneckerPilotPattern | 13 | 39 | True | True |
| kronecker_two_pilot_symbols | KroneckerPilotPattern | 26 | 26 | True | True |
