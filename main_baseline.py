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


# path = '/app/traj.pkl'
# dataset = AirportTrajData(path, num_in=60, num_out=60)

# with open('/app/trajdataset3d.pkl', 'wb') as f:
#     pickle.dump(dataset, f)


with open('/app/trajdataset3d.pkl', 'rb') as f:
    dataset = pickle.load(f)

np.random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_input', type=int, default=3)
parser.add_argument('--num_features', type=int, default=3)
parser.add_argument('--num_pred', type=int, default=60)
parser.add_argument('--input_length', type=int, default=60)
parser.add_argument('--emb', type=int, default=128)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--encoding_type', type=str, default="absdev", choices=["dev", "abs", "absdev"])

arg = parser.parse_args()

batch_size = arg.batch_size
num_input = arg.num_input
num_features = arg.num_features
num_pred = arg.num_pred
input_length = arg.input_length
emb = arg.emb
hidden = arg.hidden
alpha = arg.alpha
encoding_type = arg.encoding_type

# num_input = 3
# num_features = 3
# num_pred = 60
# input_length = 60
# emb = 128
# hidden = 64
# alpha = 0.01
# encoding_type = "dev"
# [x.shape for x in dataset.train_data]

train_data = np.stack(dataset.train_data, 0).astype(np.float32)
val_data = np.stack(dataset.val_data, 0).astype(np.float32)
test_data = np.stack(dataset.test_data, 0).astype(np.float32)


# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.scatter(test_data[:, :, 1], test_data[:, :, 2], s=1)

# (train_data[:, :, 4] > 0.5).sum()


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
    encoding_type=encoding_type,
    hidden=hidden,
)


optim = torch.optim.Adam(norm_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
total_loss = []

fig = plt.figure(figsize=(20, 20))

logdir = f"./logs/RNNMLP_{datetime.datetime.now()}_{encoding_type}_input{num_input}_pred{num_pred}_hidden{hidden}_emb{emb}_alpha{alpha}"
writer = SummaryWriter(logdir)

# create directory "./model" for saving model
if not os.path.exists(f"{logdir}/model"):
    os.makedirs(f"{logdir}/model")
if not os.path.exists(f"{logdir}/fig"):
    os.makedirs(f"{logdir}/fig")

for epoch in range(100):
    norm_model.train()

    losses = 0
    sample_mae_sum = 0
    sample_mae_sum_x = 0
    sample_mae_sum_y = 0
    sample_mae_sum_z = 0
    cnt = 0
    for i, data in (pbar := tqdm(enumerate(train_loader), total=train_loader.__len__())):
        optim.zero_grad()
        data = data.to(device)
        input = data[:, :input_length, [1, 2, 4]]
        target = data[:, input_length:, [1, 2, 4]]

        input = scaler.transform(input)
        target = scaler.transform(target)

        nllloss = norm_model.log_prob(target, input)

        nllloss = -nllloss.mean()
        loss = nllloss

        loss.backward()
        optim.step()
        losses += loss.item()

        writer.add_scalar("train_instance/nllloss", nllloss.item(), epoch * train_loader.__len__() + i)

        cnt += 1
        pbar.set_description(
            f"Train Epoch {epoch} | Loss {loss.item():.4f} | MAE {loss.item():.4f} ")

        if i == train_loader.__len__()-1:
            total_loss.append(losses/cnt)
            writer.add_scalar("train/loss", losses/cnt, epoch)

            pbar.set_description(f"Train Epoch {epoch} | Loss {loss.item():.4f} ")

    scheduler.step()
    # save
    torch.save(norm_model.state_dict(), f"{logdir}/model/model_{epoch}.pt")

    norm_model.eval()
    N = 30

    if not os.path.exists(f"{logdir}/fig/{epoch}"):
        os.makedirs(f"{logdir}/fig/{epoch}")

    losses = 0
    sample_mae_sum = 0
    sample_mae_sum_x = 0
    sample_mae_sum_y = 0
    sample_mae_sum_z = 0
    cnt = 0

    best_performance = 100000

    with torch.no_grad():
        for i, data in (pbar := tqdm(enumerate(val_loader), total=val_loader.__len__())):
            data = data.to(device)
            input = data[:, :input_length, [1, 2, 4]]
            target = data[:, input_length:, [1, 2, 4]]

            input = scaler.transform(input)
            target = scaler.transform(target)

            nllloss = norm_model.log_prob(target, input)

            nllloss = -nllloss.mean()
            loss = nllloss

            losses += loss.item()

            cnt += 1
            pbar.set_description(
                f"Val Epoch {epoch} | Loss {loss.item():.4f} ")

            if i == val_loader.__len__()-1:
                total_loss.append(losses/cnt)
                writer.add_scalar("val/loss", losses/cnt, epoch)

                pbar.set_description(f"Val Epoch {epoch} | Loss {loss.item():.4f} ")

                if best_performance > losses/cnt:
                    best_performance = losses/cnt
                    torch.save(norm_model.state_dict(), f"{logdir}/model/model_best.pt")

norm_model.load_state_dict(torch.load(f"{logdir}/model/model_best.pt"))
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
