from itertools import cycle

import numpy as np


class SequenceStratifiedKFold:
    """
    Partitions a data set of sequence labels and classifications into 10 stratified folds.
    See Dietterich, 1997 "Approximate Statistical Tests for Comparing Supervised Classification
    Algorithms" for in-depth analysis.

    Each partition should have an evenly distributed representation of sequence labels.
    Without stratification, under-representated labels may not appear in some folds.
    """
    def __init__(self, folds=10):
        self.folds = folds

    def __call__(self, X, y):
        """
        Returns an iterable [(X*,y*), ...] where each element contains the indices
        of the train and test set for the particular testing fold.

        :param X: a collection of sequences
        :param y: a collection of sequence labels
        :return:
        """

        # labels are ordered by most examples in data
        labels = np.unique([label for sequence in y for label in sequence])
        np.flip(labels)

        added = np.ones(len(y), dtype=bool)
        partitions = [[] for fold in range(self.folds)]
        partition_cycler = cycle(partitions)

        for label in labels:
            possible_sequences = [index for index, sequence in enumerate(y) if label in sequence]
            for index in possible_sequences:
                if added[index]:
                    partition = next(partition_cycler)
                    partition.append(index)
                    added[index] = 0
        train_test_array = []

        for i, _ in enumerate(partitions):
            y = partitions[i]
            X = []
            for j, partition in enumerate(partitions):
                if i != j:
                    X += partition

            # print(X)
            train_test_array.append((X,y))

        return train_test_array
