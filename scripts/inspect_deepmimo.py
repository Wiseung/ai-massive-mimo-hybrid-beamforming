#!/usr/bin/env python
"""Inspect local or downloadable DeepMIMO availability."""

from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.deepmimo_loader import load_deepmimo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--scenario-path", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--bs-idx", type=int, default=0)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--num-bs-ant", type=int, default=None)
    parser.add_argument("--num-subcarriers", type=int, default=None)
    parser.add_argument("--subcarrier-idx", type=int, default=None)
    parser.add_argument("--narrowband", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        dataset = load_deepmimo_dataset(
            scenario_path=args.scenario_path,
            scenario=args.scenario,
            download=args.download,
            bs_idx=args.bs_idx,
            num_users=args.num_users,
            num_bs_ant=args.num_bs_ant,
            num_subcarriers=args.num_subcarriers,
            subcarrier_idx=args.subcarrier_idx,
            narrowband=args.narrowband or args.num_subcarriers in (None, 1),
        )
        print("Loaded DeepMIMO dataset.")
        print("Length:", len(dataset))
        print("Metadata:", dataset.metadata)
        print("Channel shape:", tuple(dataset.channels.shape))
    except ImportError as exc:
        print(f"DeepMIMO inspection failed: {exc}")
    except Exception as exc:
        print(f"DeepMIMO inspection failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
