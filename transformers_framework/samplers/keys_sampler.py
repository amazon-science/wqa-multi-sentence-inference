import math
from typing import Dict, Iterator, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler, T_co


def get_keys_to_indexes(dataset: Dataset) -> Dict:
    r""" Returns a dict mapping keys to tuples (key, index). """
    keys_to_indexes = {}

    for i in range(len(dataset)):
        sample = dataset._get_sample(i)
        if sample.key not in keys_to_indexes:
            keys_to_indexes[sample.key] = []
        keys_to_indexes[sample.key].append((sample.key, i))

    return keys_to_indexes


def get_indexes(
    dataset: Dataset,
    generator: torch.Generator,
    shuffle: bool = True,
):
    r""" Simply iterate over the keys and provide all the data in a shuffled-at-the-key-level order. """

    keys_to_indexes = get_keys_to_indexes(dataset)
    if shuffle is True:
        permutation = torch.randperm(len(keys_to_indexes), generator=generator).tolist()
    else:
        permutation = torch.arange(len(keys_to_indexes)).tolist()
    keys_indexes = [x for i in permutation for x in keys_to_indexes[i]]

    # simply yield every index
    for triple in keys_indexes:
        yield triple[1]


class KeysSampler(Sampler[T_co]):
    r"""Samples elements randomly by providing elements with the same key sequentially.

    Args:
        dataset (Dataset): dataset to sample from
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        dataset: Dataset,
        generator: torch.Generator = None,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.generator = generator
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[T_co]:
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        yield from get_indexes(
            dataset=self.dataset,
            generator=generator,
            shuffle=self.shuffle,
        )

    def __len__(self):
        return len(self.dataset)


class DistributedKeysSampler(DistributedSampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the indexes returned by KeysSampler.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        generator (Generator): Generator used in sampling.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )

    def __iter__(self) -> Iterator[T_co]:

        # deterministically shuffle based on epoch and seed
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # shuffle at the key level, not at the index one
        indices = get_indexes(
            dataset=self.dataset,
            generator=generator,
            shuffle=self.shuffle,
        )
        indices = list(indices)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples:(self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
