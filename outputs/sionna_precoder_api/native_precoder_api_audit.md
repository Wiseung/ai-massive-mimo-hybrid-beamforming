# Sionna Native Precoder API Audit

- sionna_import_ok: `True`
- sionna_version: `2.0.1`
- sionna_rzf_precoder_available: `True`
- recommended_next_step: `adapter_bridge`

| Target | Import OK | Constructor | Call | Input shape | Output shape | Probe OK |
| --- | --- | --- | --- | --- | --- | --- |
| RZFPrecoder | True | `(resource_grid: 'sionna.phy.ofdm.ResourceGrid', stream_management: 'sionna.phy.mimo.StreamManagement', return_effective_channel: bool = False, precision: Optional[Literal['single', 'double']] = None, device: Optional[str] = None, **kwargs) -> None` | `(self, x: torch.Tensor, h: torch.Tensor, alpha: Union[float, torch.Tensor] = 0.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]` | [B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] plus h=[B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size] | [B, num_tx, num_tx_ant, num_ofdm_symbols, fft_size] plus optional h_eff | True |
| PrecodedChannel | True | `(resource_grid: 'sionna.phy.ofdm.ResourceGrid', stream_management: 'sionna.phy.mimo.StreamManagement', precision: Optional[Literal['single', 'double']] = None, device: Optional[str] = None, **kwargs) -> None` | `(self, h: torch.Tensor, tx_power: torch.Tensor, h_hat: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor` | h=[B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tx_power=[B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] | h_eff=[B, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers] | False |
| StreamManagement | True | `(rx_tx_association: numpy.ndarray, num_streams_per_tx: int) -> None` | `unavailable: AttributeError: type object 'StreamManagement' has no attribute 'call'` | rx_tx_association ndarray plus num_streams_per_tx | stream-management object | False |

## Summary
1. Sionna RZFPrecoder usable: `True`
2. expected input shape: `[B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] plus h=[B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]`
3. expected output shape: `[B, num_tx, num_tx_ant, num_ofdm_symbols, fft_size] plus optional h_eff`
4. compatible with ExtractedCSI / PrecoderOutput: `True`
5. recommendation: `adapter_bridge`

## Notes
- RZFPrecoder exists in Sionna 2.0.1 and is callable on the current install.
- Its native input contract is resource-grid-centric and higher rank than the project's H_f=(B,Nsc,K,Nt) path.
- Current adapter can map one ExtractedCSI object into a probe-only native call path and convert the native output into PrecoderOutput.
- This does not replace the project-side precoder mainline and does not justify a full native-only benchmark claim.
