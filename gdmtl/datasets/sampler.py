from __future__ import annotations

from typing import Iterator, List

import numpy as np
from more_itertools import chunked
from torch.utils.data import Sampler

from .rank_dataset import TokenCountDataset


class DynamicBatchSampler(Sampler):  # type:ignore
    def __init__(
        self,
        dataset: TokenCountDataset,
        batch_tokens: int,
        mod: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_tokens = batch_tokens
        self.batches: List[List[int]] = []
        self.mod = mod

        batch: List[int] = [0]
        size = self.dataset.estimate_tokens(0)
        total_size = size
        for idx in range(1, len(self.dataset)):
            if total_size + size > self.batch_tokens:
                bsz = len(batch) // self.mod * self.mod
                self.batches.append(batch[:bsz])

                batch = batch[bsz:] + [idx]
                # Only estimate the size for the first example of a batch. Following
                # will be padded to the same size.
                size = self.dataset.estimate_tokens(batch[0])
                total_size = size * len(batch)
            else:
                batch.append(idx)
                total_size += size
        if batch:
            self.batches.append(batch)
        if shuffle:
            np.random.shuffle(self.batches)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def avg_bsz(self) -> float:
        size: float = np.mean([len(x) for x in self.batches])
        return size


class FixedBatchSampler(Sampler):  # type:ignore
    """This batch sample is identical to the default pytorch BatchSampler except that it
    supports post-sampling shuffle. This is useful in the case that the dataset needs to
    be sorted by length and thus the shuffling must be delayed here.

    """

    def __init__(
        self, dataset: TokenCountDataset, batch_size: int, shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = list(chunked(list(range(len(self.dataset))), batch_size))
        if shuffle:
            np.random.shuffle(self.batches)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)
