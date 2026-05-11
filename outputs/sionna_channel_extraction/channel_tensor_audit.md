# Sionna Channel Tensor Audit

- Sionna import ok: `True`
- Sionna version: `2.0.1`
- Recommended extraction path: `OFDMChannel(return_channel=True) -> extract_h_f_from_sionna_channel`

| Component | Import OK | Constructor | Call | Channel Returned | Output Shape | Probe OK |
| --- | --- | --- | --- | --- | --- | --- |
| OFDMChannel | True | `(channel_model: sionna.phy.channel.channel_model.ChannelModel, resource_grid: sionna.phy.ofdm.resource_grid.ResourceGrid, normalize_channel: bool = False, return_channel: bool = False, precision: Optional[Literal['single', 'double']] = None, device: Optional[str] = None, **kwargs) -> None` | `(self, x: torch.Tensor, no: Union[float, torch.Tensor, NoneType] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]` | True | [2, 4, 1, 2, 19] | True |
| ApplyOFDMChannel | True | `(precision: Optional[Literal['single', 'double']] = None, device: Optional[str] = None, **kwargs) -> None` | `(self, x: torch.Tensor, h_freq: torch.Tensor, no: Union[float, torch.Tensor, NoneType] = None) -> torch.Tensor` | False | None | False |
| RayleighBlockFading | True | `(num_rx: int, num_rx_ant: int, num_tx: int, num_tx_ant: int, precision: Optional[str] = None, device: Optional[str] = None, **kwargs) -> None` | `` | False | None | False |
| GenerateOFDMChannel | True | `(channel_model: sionna.phy.channel.channel_model.ChannelModel, resource_grid: sionna.phy.ofdm.resource_grid.ResourceGrid, normalize_channel: bool = False, precision: Optional[Literal['single', 'double']] = None, device: Optional[str] = None, **kwargs) -> None` | `` | False | None | False |
| cir_to_ofdm_channel | True | `(frequencies: torch.Tensor, a: torch.Tensor, tau: torch.Tensor, normalize: bool = False) -> torch.Tensor` | `` | False | None | False |
| subcarrier_frequencies | True | `(num_subcarriers: int, subcarrier_spacing: float, precision: Optional[str] = None, device: Optional[str] = None) -> torch.Tensor` | `` | False | None | False |

## Summary
- OFDMChannel returns channel tensor: `True`
- observed channel tensor shape: `[2, 4, 1, 1, 16, 2, 19]`
- project H_f conversion possible: `True`
- project H_f shape: `[2, 16, 4, 16]`
- Current observed OFDMChannel channel tensor axes are interpreted as batch/rx/rx_ant/tx/tx_ant/ofdm_symbol/fft_bin.
- The current bridge assumes num_tx=1, rx=user, rx_ant=1, and tx_ant=Nt for MU downlink extraction.
- ResourceGrid metadata for the probe: {'fallback_used': False, 'fallback_reason': '', 'num_users': 4, 'num_tx': 1, 'num_streams_per_tx': 4, 'fft_size': 19, 'num_data_symbols': 16, 'num_pilot_symbols': 16, 'effective_subcarrier_ind': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17], 'rx_tx_association': [[1], [1], [1], [1]]}
