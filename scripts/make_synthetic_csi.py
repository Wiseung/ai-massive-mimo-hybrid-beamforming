#!/usr/bin/env python
"""Generate synthetic CSI datasets."""

from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import save_channel_dataset
from beamforming.data.synthetic import (
    rayleigh_narrowband_channel,
    sparse_geometric_mmwave_channel,
    wideband_ofdm_channel,
)
from beamforming.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--num-bs-ant", type=int, default=64)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--num-rf-chains", type=int, default=4)
    parser.add_argument("--num-paths", type=int, default=3)
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel-type", choices=["rayleigh", "mmwave", "wideband"], default="mmwave")
    parser.add_argument("--num-subcarriers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.channel_type == "rayleigh":
        channels = rayleigh_narrowband_channel(
            num_samples=args.num_samples,
            num_users=args.num_users,
            num_bs_ant=args.num_bs_ant,
        )
    elif args.channel_type == "mmwave":
        channels = sparse_geometric_mmwave_channel(
            num_samples=args.num_samples,
            num_users=args.num_users,
            num_bs_ant=args.num_bs_ant,
            num_paths=args.num_paths,
        )
    else:
        channels = wideband_ofdm_channel(
            num_samples=args.num_samples,
            num_users=args.num_users,
            num_bs_ant=args.num_bs_ant,
            num_paths=args.num_paths,
            num_subcarriers=args.num_subcarriers,
        )
    metadata = {
        "channel_type": args.channel_type,
        "num_samples": args.num_samples,
        "num_bs_ant": args.num_bs_ant,
        "num_users": args.num_users,
        "num_rf_chains": args.num_rf_chains,
        "num_paths": args.num_paths,
        "seed": args.seed,
    }
    save_channel_dataset(args.out, channels, args.snr_list, metadata)
    print(f"Saved synthetic dataset to {args.out}")
    print("Channel shape:", tuple(channels.shape))


if __name__ == "__main__":
    main()
