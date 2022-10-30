from pickle import TRUE
from numpy import testing
import torch
import numpy as np

from src.util import normalization, sequence_data, addSOSEOS, feat_normalization
from torch.utils.data import Dataset, DataLoader


class DataManager(object):
    def __init__(self, input, features, boundaries, feature_boundary, train_ratio, batch_size):
        self.raw_input = input
        self.features = features

        self.boundaries = boundaries
        self.raw_input = normalization(
            self.raw_input, boundaries[0], boundaries[1], boundaries[2], boundaries[3])

        self.feature_boundary = feature_boundary
        # normalize features
        self.features = feat_normalization(self.features, feature_boundary)

        self.train_ratio = train_ratio
        self.test_ratio = 1-train_ratio

        n_data = self.raw_input.shape[0]

        randidx = np.random.permutation(n_data)
        # randidx = np.random.permutation(n_data)
        n_data = len(randidx)
        n_train = int(n_data*train_ratio)
        n_test = n_data-n_train

        self.train_idx = randidx[:n_train]
        train_idx = self.train_idx
        test_idx = randidx[n_train:(n_train+n_test)]

        self.train_input = torch.FloatTensor(self.raw_input[train_idx])
        self.test_input = torch.FloatTensor(self.raw_input[test_idx])

        self.train_feature = torch.FloatTensor(self.features[train_idx])
        self.test_feature = torch.FloatTensor(self.features[test_idx])

        self.train_data = sequence_data(self.train_input, self.train_feature)
        self.test_data = sequence_data(self.test_input, self.test_feature)

        # loaders load input and feature together
        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
