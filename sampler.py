import torch
import torch.nn.functional as F
import torch.utils.data
import random
import numpy as np

from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_groups.
    Returns batches of size n_groups * (batch_size // n_groups)
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, group_labels, batch_size):
        groups = sorted(set(group_labels.numpy()))
        print(groups)

        n_groups = len(groups)
        self._n_samples = batch_size // n_groups
        if self._n_samples == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of groups, got {batch_size}"
            )

        self._group_iters = [
            InfiniteSliceIterator(np.where(group_labels == group_)[0], group_=group_)
            for group_ in groups
        ]

        batch_size = self._n_samples * n_groups
        self.n_dataset = len(group_labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {batch_size}"
            )
        print("K=", n_groups, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for group_iter in self._group_iters:
                indices.extend(group_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices

        for group_iter in self._group_iters:
            group_iter.reset()

    def __len__(self):
        return self._n_batches
    
    
class InfiniteSliceIterator:
    def __init__(self, array, group_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.group_ = group_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.group_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]