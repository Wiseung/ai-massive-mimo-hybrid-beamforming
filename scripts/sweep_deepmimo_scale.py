#!/usr/bin/env python
"""Report which DeepMIMO scales are actually available from a saved tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_channel_dataset(args.data)
    channels = dataset.channels
    metadata = dataset.metadata
    available_users = int(metadata.get("num_users", channels.shape[-2]))
    available_ant = int(metadata.get("num_bs_ant", channels.shape[-1]))
    available_sub = int(metadata.get("num_subcarriers", 1 if channels.ndim == 3 else channels.shape[-2]))

    rows: list[dict[str, object]] = []
    for num_users in (2, 4, 8):
        for num_bs_ant in (8, 16, 32):
            for num_subcarriers in (1, 8, 16):
                is_available = num_users <= available_users and num_bs_ant <= available_ant and num_subcarriers <= available_sub
                reason = "available"
                if not is_available:
                    blockers = []
                    if num_users > available_users:
                        blockers.append(f"K>{available_users}")
                    if num_bs_ant > available_ant:
                        blockers.append(f"Nt>{available_ant}")
                    if num_subcarriers > available_sub:
                        blockers.append(f"Nsc>{available_sub}")
                    reason = "unavailable: " + ", ".join(blockers)
                rows.append(
                    {
                        "requested_num_users": num_users,
                        "requested_num_bs_ant": num_bs_ant,
                        "requested_num_subcarriers": num_subcarriers,
                        "available": is_available,
                        "reason": reason,
                        "available_num_users": available_users,
                        "available_num_bs_ant": available_ant,
                        "available_num_subcarriers": available_sub,
                        "narrowband_only_tensor": available_sub == 1,
                    }
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved DeepMIMO available scale table to {out_path}")
    if available_ant < 32 or available_sub < 16:
        print(
            "Current tensor cannot support the full requested sweep. "
            f"Available tensor limits: K={available_users}, Nt={available_ant}, Nsc={available_sub}."
        )


if __name__ == "__main__":
    main()
