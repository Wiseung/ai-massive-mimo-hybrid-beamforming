from __future__ import annotations

from beamforming.data.splits import make_dataset_split, split_counts


def test_random_split_counts_sum_to_total() -> None:
    payload = make_dataset_split(100, split_mode="random", seed=42)
    counts = split_counts(payload)
    assert counts["train"] + counts["val"] + counts["test"] == 100


def test_contiguous_split_is_ordered() -> None:
    payload = make_dataset_split(20, split_mode="contiguous", seed=0)
    assert payload["train_indices"] == list(range(14))
    assert payload["val_indices"] == list(range(14, 17))
    assert payload["test_indices"] == list(range(17, 20))
