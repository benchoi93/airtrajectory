from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import pdb
import pickle
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d


class AirportTrajData(object):
    def __init__(self, path, num_in=60, num_out=60):
        self.path = path

        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)

        self.num_in = num_in
        self.num_out = num_out

        train_ratio = 0.7
        val_ratio = 0.1
        test_ratio = 0.2

        # random shuffle self.data
        # self.data = np.random.shuffle(self.data)

        # split data
        self.train_data = self.data[:int(len(self.data) * train_ratio)]
        self.val_data = self.data[int(len(self.data) * train_ratio):int(len(self.data) * (train_ratio + val_ratio))]
        self.test_data = self.data[int(len(self.data) * (train_ratio + val_ratio)):]

        self.train_data = self.list_to_tensor(self.train_data)
        self.val_data = self.list_to_tensor(self.val_data)
        self.test_data = self.list_to_tensor(self.test_data)

    def list_to_tensor(self, data):
        # data : list of pandas

        temp = []
        # iterate through data
        # and generate trajectory with length (num_in + num_out)

        for i in tqdm(range(len(data))):
            # iterate through each trajectory
            for j in range(len(data[i]) - (self.num_in + self.num_out)):
                out = np.zeros((self.num_in + self.num_out+1, 7))

                out_temp = data[i][j:j + self.num_in + self.num_out+1]
                out_temp = out_temp[out_temp['tod'].between(out_temp['tod'].iloc[0], out_temp['tod'].iloc[0]+self.num_in+self.num_out)]

                out[out_temp['tod'] - out_temp['tod'].iloc[0], :5] = out_temp[['tod', 'lon', 'lat', 'alt', 'fl']].values

                out[out[:, 4] < 5] = 0

                if (out[:, 0] == 0).sum() / out.shape[0] < 0.1:

                    out[:, 1] = interp1d(np.where(out[:, 1] != 0)[0], out[np.where(out[:, 1] != 0)[0], 1],
                                         fill_value="extrapolate")(np.arange(out.shape[0]))
                    out[:, 2] = interp1d(np.where(out[:, 2] != 0)[0], out[np.where(out[:, 2] != 0)[0], 2],
                                         fill_value="extrapolate")(np.arange(out.shape[0]))
                    out[:, 4] = interp1d(np.where(out[:, 4] != 0)[0], out[np.where(out[:, 4] != 0)[0], 4],
                                         fill_value="extrapolate")(np.arange(out.shape[0]))

                    # temp.append(out)

                    out[1:, -2] = out[1:, 1] - out[:-1, 1]
                    out[1:, -1] = out[1:, 2] - out[:-1, 2]

                    out = out[1:, :]

                    assert(np.all(out[:, 1] != 0))
                    temp.append(out)

        return temp



class AirportTrajTestData(AirportTrajData):
    def __init__(self, path, num_in=60, num_out=60):
        self.path = path

        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)

        self.num_in = num_in
        self.num_out = num_out

        self.test_data = self.data
        self.test_data = self.list_to_tensor(self.test_data)


class MinMaxScaler():
    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z

    def transform(self, x):
        x[..., 0] = (x[...,  0] - self.min_x) / (self.max_x - self.min_x)
        x[..., 1] = (x[..., 1] - self.min_y) / (self.max_y - self.min_y)
        x[..., 2] = (x[..., 2] - self.min_z) / (self.max_z - self.min_z)

        return x

    def inverse_transform(self, x):
        x[...,  0] = x[...,  0] * (self.max_x - self.min_x) + self.min_x
        x[...,  1] = x[..., 1] * (self.max_y - self.min_y) + self.min_y
        x[..., 2] = x[..., 2] * (self.max_z - self.min_z) + self.min_z

        return x


class GaussianScaler():
    def __init__(self, data):
        # data : size = [B, T , 2]
        self.mean = np.mean(data, axis=(0, 1))
        self.std = np.std(data, axis=(0, 1))

    def transform(self, x):
        x[..., 0] = (x[..., 0] - self.mean[0]) / self.std[0]
        x[..., 1] = (x[..., 1] - self.mean[1]) / self.std[1]

        return x

    def inverse_transform(self, x):
        x[..., 0] = x[..., 0] * self.std[0] + self.mean[0]
        x[..., 1] = x[..., 1] * self.std[1] + self.mean[1]

        return x


def gauss2d(mu, sigma, to_plot=False):
    w, h = 100, 100

    std = [np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1])]
    x = np.linspace(mu[0] - 3 * std[0], mu[0] + 3 * std[0], w)
    y = np.linspace(mu[1] - 3 * std[1], mu[1] + 3 * std[1], h)

    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(w, h, order='F')

    if to_plot:
        plt.contourf(x, y, z.T)
        plt.show()

    return z
