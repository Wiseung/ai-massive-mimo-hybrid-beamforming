#!/usr/bin/env python
"""Inspect local DeepMIMO availability and expected dataset path."""

from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.deepmimo_loader import load_deepmimo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-path", required=True)
    parser.add_argument("--bs-idx", type=int, default=0)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--subcarrier-idx", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        dataset = load_deepmimo_dataset(
            args.scenario_path,
            bs_idx=args.bs_idx,
            num_users=args.num_users,
            subcarrier_idx=args.subcarrier_idx,
        )
        print("Loaded DeepMIMO dataset.")
        print("Length:", len(dataset))
        print("Metadata:", dataset.metadata)
    except Exception as exc:
        print(f"DeepMIMO inspection failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
