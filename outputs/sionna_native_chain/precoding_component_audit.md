# Sionna Precoding Component Audit

- Sionna import ok: `True`
- Sionna version: `2.0.1`
- Recommendation: `use project frequency-domain precoder insertion first; keep Sionna RZFPrecoder as an optional shape-checked reference path`

| Component | Import OK | Constructor | Call | Expected Input | Expected Output | Probe OK |
| --- | --- | --- | --- | --- | --- | --- |
| RZFPrecoder | True | `(resource_grid: 'sionna.phy.ofdm.ResourceGrid', stream_management: 'sionna.phy.mimo.StreamManagement', return_effective_channel: bool = False, precision: Optional[Literal['single', 'double']] = None, device: Optional[str] = None, **kwargs) -> None` | `(self, x: torch.Tensor, h: torch.Tensor, alpha: Union[float, torch.Tensor] = 0.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]` | [B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] plus h with Sionna channel tensor layout | precoded resource grid, optionally effective channel | True |
| PrecodedChannel | True | `(resource_grid: 'sionna.phy.ofdm.ResourceGrid', stream_management: 'sionna.phy.mimo.StreamManagement', precision: Optional[Literal['single', 'double']] = None, device: Optional[str] = None, **kwargs) -> None` | `(self, h: torch.Tensor, tx_power: torch.Tensor, h_hat: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor` | effective channel tensor h plus tx_power, optional h_hat | effective post-precoding channel | False |
| StreamManagement | True | `(rx_tx_association: numpy.ndarray, num_streams_per_tx: int) -> None` | `unavailable: AttributeError: type object 'StreamManagement' has no attribute 'call'` | rx_tx_association ndarray, num_streams_per_tx | stream-management object | True |

## Notes
- Current project channel/precoder tensors use H_f=(B, Nsc, K, Nt) and F_f=(B, Nsc, Nt, K).
- Current Sionna RZFPrecoder expects a resource-grid tensor plus a higher-rank channel tensor layout, so direct substitution is not the clean mainline path.
