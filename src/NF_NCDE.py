import torchcde
import copy
import math

from src.ncde.NCDE import NeuralCDE

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class rolling_encoder(nn.Module):
    def __init__(self, input_size, emb_dim, hidden, feat_size):
        super().__init__()
        self.input_size = input_size
        self.feat_size = feat_size
        self.hidden = hidden
        self.emb_dim = emb_dim

        # fc layer with 3 layers and relu activation
        self.emb_fc = nn.Sequential(
            nn.Linear(input_size + feat_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

        # self.emb_fc = nn.Linear(input_size, emb_dim)

        self.gru = nn.GRU(self.emb_dim, self.hidden, num_layers=3, batch_first=True)

    def forward(self, x, feat):
        # x: (batch, seq_len, input_size)
        # feat: (batch, seq_len, feat_size)
        batch_size, seq_len, _ = x.shape

        # concat x and feat
        x = torch.cat([x, feat], dim=-1)

        x = self.emb_fc(x)
        #h_0 = self._init_state(batch_size = x.size(0))
        x, _ = self.gru(x)

        return x[:, -1, :]


class normalizing_flow(nn.Module):
    def __init__(self, input_size, feat_size, emb_dim, hidden, n_blocks, num_flow_layers, num_pred, cond_label_size=None, flow_model="real_nvp"):
        super().__init__()
        # self.gru_model = rolling_encoder(input_size, emb_dim, hidden, feat_size)
        self.ncde = NeuralCDE(input_channels=input_size+feat_size,
                              hidden_channels=8,
                              output_channels=hidden)

        # self.flow = norm_model
        if flow_model == "real_nvp":
            from src.flow.realNVP import RealNVP
            self.flow = RealNVP(n_blocks=n_blocks,
                                input_size=input_size,
                                hidden_size=hidden,
                                n_hidden=num_flow_layers,
                                num_pred=num_pred,
                                cond_label_size=cond_label_size,
                                batch_norm=True
                                )
        else:
            raise NotImplementedError

    def get_condition(self, x, feat):
        # condition encoder based on Neural CDE
        # x: (batch, seq_len, input_size)
        # feat: (batch, seq_len, feat_size)
        input = torch.cat([x, feat], dim=-1)
        input_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(input)
        output = self.ncde(input_coeff)

        return output

    def forward(self, x, y, feat):
        enroll_rnn_out = self.get_condition(x, feat)
        output = self.flow.log_prob(y, enroll_rnn_out)
        #output = self.flow.log_prob(y,cond=None)
        return -output
