from tensorboardX import SummaryWriter
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import MinMaxScaler, GaussianScaler
from src.FloMoRNN import FloMo
import torch
import numpy as np
import sched
import pickle
from util import AirportTrajData
import datetime
import pandas as pd

from math import sin, cos, sqrt, atan2, radians

with open('/app/trajdataset3d.pkl', 'rb') as f:
    dataset = pickle.load(f)

np.random.seed(0)
torch.manual_seed(0)

logdir = "RNNMLP_2022-12-15 23:44:58.390426_absdev_input3_pred60_hidden64_emb128_alpha0.01"
# logdir = "RNNMLP_2022-12-15 23:53:58.661212_abs_input3_pred60_hidden64_emb128_alpha0.01"
# logdir = "RNNMLP_2022-12-15 23:53:54.437726_dev_input3_pred60_hidden64_emb128_alpha0.01"

model, logtime, encoding_type, input, pred, hidden, emb, alpha = logdir.split("_")

batch_size = 64
num_input = int(input.split("input")[1])
num_features = 0
num_pred = int(pred.split("pred")[1])
input_length = 60
emb = int(emb.split("emb")[1])
hidden = int(hidden.split("hidden")[1])
alpha = float(alpha.split("alpha")[1])
encoding_type = encoding_type

train_data = np.stack(dataset.train_data, 0).astype(np.float32)
val_data = np.stack(dataset.val_data, 0).astype(np.float32)
test_data = np.stack(dataset.test_data, 0).astype(np.float32)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

gaussian_scaler = GaussianScaler(train_data[:, :, -2:])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_x = train_data[:, :, 1].min()
max_x = train_data[:, :, 1].max()
min_y = train_data[:, :, 2].min()
max_y = train_data[:, :, 2].max()
# min_z = train_data[:, :, 4].min()
min_z = 5
# max_z = train_data[:, :, 4].max()
max_z = 200

train_data[:, :, 4][train_data[:, :, 4] < 5] = min_z
train_data[:, :, 4][train_data[:, :, 4] > 200] = max_z

# define a min-max scaler
scaler = MinMaxScaler(min_x, max_x, min_y, max_y, min_z, max_z)

norm_model = FloMo(
    hist_size=input_length,
    pred_steps=num_pred,
    alpha=3,
    beta=0.002,
    gamma=0.002,
    num_in=num_input,
    encoding_type=encoding_type
)

norm_model.load_state_dict(torch.load(os.path.join("/app/logs", logdir, "model", "model_best.pt")))
norm_model.eval()

performance = np.zeros((len(test_loader) * batch_size, 12))
cnt = 0
with torch.no_grad():
    for i, data in (pbar := tqdm(enumerate(test_loader), total=test_loader.__len__())):
        data = data.to(device)
        input = data[:, :input_length, [1, 2, 4]]
        target = data[:, input_length:, [1, 2, 4]]

        input = scaler.transform(input)
        target = scaler.transform(target)

        samples = norm_model.predict(input)
        samples = samples.unsqueeze(1)

        samples = scaler.inverse_transform(samples)
        target = scaler.inverse_transform(target)

        # calculate minADE
        dlon = torch.deg2rad(samples[:, :, :, 0]) - torch.deg2rad(target[:, :, 0].unsqueeze(1))
        dlat = torch.deg2rad(samples[:, :, :, 1]) - torch.deg2rad(target[:, :, 1].unsqueeze(1))

        a = torch.sin(dlat / 2)**2 + torch.cos(torch.deg2rad(target[:, :, 1].unsqueeze(1))) * \
            torch.cos(torch.deg2rad(samples[:, :, :, 1])) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        dist_xy = 6371 * c
        dz = samples[:, :, :, 2] - target[:, :, 2].unsqueeze(1)
        dalt = dz * 1000 * 0.3048

        dist = torch.sqrt(dist_xy**2 + dalt**2)

        performance[(cnt):(cnt+samples.shape[0]), 0] = dist.mean(-1).min(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 1] = dist.mean(-1).max(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 2] = dist.mean(-1).mean(-1).cpu().numpy()

        # calculate minFDE
        performance[(cnt):(cnt+samples.shape[0]), 3] = dist[..., -1].min(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 4] = dist[..., -1].max(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 5] = dist[..., -1].mean(-1).cpu().numpy()

        cnt += samples.shape[0]

print(f"minADE: {performance[:, 0].mean():.4f}")
print(f"maxADE: {performance[:, 1].mean():.4f}")
print(f"meanADE: {performance[:, 2].mean():.4f}")

print(f"minFDE: {performance[:, 3].mean():.4f}")
print(f"maxFDE: {performance[:, 4].mean():.4f}")
print(f"meanFDE: {performance[:, 5].mean():.4f}")
