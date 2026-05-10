"""Scaffold for a future WMMSE baseline.

The repository does not expose this baseline through `run_baselines.py` yet
because a validated MU-downlink WMMSE implementation has not been completed.
"""

from __future__ import annotations

import torch


def wmmse_precoder(
    channel: torch.Tensor,
    noise_var: float | torch.Tensor,
    total_power: float = 1.0,
    num_iters: int = 50,
) -> torch.Tensor:
    """Placeholder for a future WMMSE solver.

    TODO:
    - implement alternating receiver / weight / precoder updates
    - validate convergence and power projection
    - add tests before enabling this method in public scripts
    """
    raise NotImplementedError(
        "WMMSE baseline is not implemented yet. The scaffold exists, but the method is intentionally disabled."
    )
