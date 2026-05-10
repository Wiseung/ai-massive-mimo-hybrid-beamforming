#!/usr/bin/env python
"""Create reproducible dataset splits for synthetic or DeepMIMO tensors."""

from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset
from beamforming.data.deepmimo_loader import load_deepmimo_dataset
from beamforming.data.splits import make_dataset_split, save_dataset_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--data", required=True)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-mode", choices=["random", "contiguous"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--narrowband", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace):
    if args.dataset_type == "deepmimo" or args.scenario is not None:
        return load_deepmimo_dataset(
            scenario_path=args.data,
            scenario=args.scenario,
            download=args.download,
            num_users=args.num_users,
            narrowband=args.narrowband or True,
        )
    return load_channel_dataset(args.data)


def main() -> None:
    args = parse_args()
    dataset = _load_dataset(args)
    split_payload = make_dataset_split(
        len(dataset),
        split_mode=args.split_mode,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    split_payload["dataset_path"] = args.data
    split_payload["dataset_type"] = args.dataset_type
    split_payload["num_samples"] = len(dataset)
    out_path = save_dataset_split(args.out, split_payload, dataset_metadata=dataset.metadata)
    print(f"Saved split to {out_path}")
    print({k: len(split_payload[f'{k}_indices']) for k in ('train', 'val', 'test')})


if __name__ == "__main__":
    main()
