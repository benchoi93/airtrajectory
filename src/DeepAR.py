import numpy as np
import math
import torch
import torch.nn as nn


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


class DeepAR(nn.Module):
    def __init__(self, input_size, feat_size, emb_dim, hidden, n_blocks, num_flow_layers, num_pred, cond_label_size=None, flow_model="real_nvp"):
        super().__init__()
        self.gru_model = rolling_encoder(input_size, emb_dim, hidden, feat_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_pred*4),
        )
        self.num_pred = num_pred

    def forward(self, x, y, feat):
        enroll_rnn_out = self.gru_model(x, feat)

        pred = self.mlp(enroll_rnn_out)

        dx_mu = pred[:, :self.num_pred*2]
        log_sigma = pred[:, self.num_pred*2:]

        dx_mu = dx_mu.view(-1, self.num_pred, 2)
        log_sigma = log_sigma.view(-1, self.num_pred, 2)

        mu = torch.cumsum(dx_mu, dim=1)
        mu = x[:, -1, :].unsqueeze(1) + mu

        # pred : (batch, num_pred, 2)
        # pred [:,:,0] : mean
        # pred [:,:,1] : std

        y_loc = y[:, :, :2]
        y_dx = y[:, :, 2:]
        # calculate log likelihood
        log_prob = -0.5 * np.log(2 * math.pi) - log_sigma - 0.5 * ((y_loc - mu) / torch.exp(log_sigma)) ** 2

        mae_dx = torch.abs(y_dx - dx_mu)
        mae_loc = torch.abs(y_loc - mu)

        info = {
            "nll": - log_prob,
            "mae_dx": mae_dx,
            "mae_loc": mae_loc,
            "dx_mu": dx_mu,
            "mu": mu,
            "log_sigma": log_sigma,
        }

        return info
