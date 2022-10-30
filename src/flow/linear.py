import copy
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from src.flow.baseflow import Flow, FlowSequential, BatchNorm


class LinearFlowLayer(nn.Module):
    def __init__(self, input_size, hidden_size, cond_label_size=None, init_sigma=0.01):
        super().__init__()

        self.init_sigma = init_sigma
        self.n_features = input_size
        self.weights = nn.Parameter(torch.randn(1, input_size).normal_(0, self.init_sigma))
        self.bias = nn.Parameter(torch.zeros(1).normal_(0, self.init_sigma))
        self.u = nn.Parameter(torch.randn(1, input_size).normal_(0, self.init_sigma))

    def forward(self, x, y=None):
        u_temp = (self.weights @ self.u.t()).squeeze()
        m_u_temp = -1 + F.softplus(u_temp)

        uhat = self.u + (m_u_temp - u_temp) * (self.weights / (self.weights @ self.weights.t()))

        z_temp = z @ self.weights.t() + self.bias  # F.linear(z, self.weights, self.bias)

        new_z = z + uhat * torch.tanh(z_temp)

        psi = (1 - torch.tanh(z_temp)**2) @ self.weights

        det_jac = 1 + psi @ uhat.t()  # uhat * psi

        logdet_jacobian = torch.log(torch.abs(det_jac) + 1e-8).squeeze()

        return new_z, logdet_jacobian


class LinearFlow(Flow):
    def __init__(self, input_size, num_pred, n_blocks, hidden_size, n_hidden, cond_label_size, activation="ReLU", batch_norm=True):
        super().__init__(input_size, num_pred)

        modules = []
        for i in range(n_blocks):
            modules += [
                LinearFlowLayer(
                    input_size, hidden_size, cond_label_size
                )
            ]
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)
