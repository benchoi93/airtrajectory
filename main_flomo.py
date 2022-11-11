from tensorboardX import SummaryWriter
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import MinMaxScaler, GaussianScaler
from src.FloMo import FloMo
import torch
import numpy as np
import sched
import pickle
from util import AirportTrajData

# path = './traj.pkl'
# dataset = AirportTrajData(path, num_in=60, num_out=60)

# with open('./trajdataset3d.pkl', 'wb') as f:
#     pickle.dump(dataset, f)


with open('./trajdataset3d.pkl', 'rb') as f:
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
parser.add_argument('--encoding_type', type=str, default="dev", choices=["dev", "abs", "absdev"])

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
    encoding_type=encoding_type
)


optim = torch.optim.Adam(norm_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
total_loss = []

fig = plt.figure(figsize=(20, 20))

logdir = f"./logs/Flomo_{encoding_type}_input{num_input}_pred{num_pred}_hidden{hidden}_emb{emb}_alpha{alpha}"
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
        # features = data[:, :input_length, [0, 5, 6]]
        target = data[:, input_length:, [1, 2, 4]]

        input = scaler.transform(input)
        target = scaler.transform(target)
        # features[:, :, 0] = features[:, :, 0] / (24*60*60)
        # features[:, :, 1:] = gaussian_scaler.transform(features[:, :, 1:])

        # add small noise to target
        # target = [target + torch.randn_like(target) * 0.0005 for _ in range(5)]
        # target = torch.concat(target, 0)
        # input = input.repeat(5, 1, 1)
        # features = features.repeat(5, 1, 1)

        nllloss = norm_model.log_prob(target, input)

        with torch.no_grad():
            samples, probs = norm_model.sample(n=10, x=input)
            sample_mae = (samples - target.unsqueeze(1)).abs()
            sample_mae_x = sample_mae[:, :, :, 0].mean()
            sample_mae_y = sample_mae[:, :, :, 1].mean()
            sample_mae_z = sample_mae[:, :, :, 2].mean()
            sample_mae = sample_mae.mean()

        nllloss = -nllloss.mean()
        # loss = alpha * nllloss + (1-alpha) * mae_loss
        loss = nllloss

        loss.backward()
        optim.step()
        losses += loss.item()
        sample_mae_sum += sample_mae.item()
        sample_mae_sum_x += sample_mae_x.item()
        sample_mae_sum_y += sample_mae_y.item()
        sample_mae_sum_z += sample_mae_z.item()
        cnt += 1
        pbar.set_description(
            f"Epoch {epoch} | Loss {loss.item():.4f} | MAE {sample_mae.item():.4f} - x {sample_mae_x.item():.4f} y {sample_mae_y.item():.4f} z {sample_mae_z.item():.4f}")
        # pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f} | NLLLoss {nllloss.item():.4f} | MAELoss {mae_loss.item()/N:.4f}")

        if i == train_loader.__len__()-1:
            total_loss.append(losses/cnt)
            writer.add_scalar("train/loss", losses/cnt, epoch)
            writer.add_scalar("train/sample_mae", sample_mae_sum/cnt, epoch)
            writer.add_scalar("train/sample_mae_x", sample_mae_sum_x/cnt, epoch)
            writer.add_scalar("train/sample_mae_y", sample_mae_sum_y/cnt, epoch)
            writer.add_scalar("train/sample_mae_z", sample_mae_sum_z/cnt, epoch)

            pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f} | MAE {sample_mae.item():.4f}")

    scheduler.step()
    # save
    torch.save(norm_model.state_dict(), f"{logdir}/model/model_{epoch}.pt")

    norm_model.eval()
    N = 30

    if not os.path.exists(f"{logdir}/fig/{epoch}"):
        os.makedirs(f"{logdir}/fig/{epoch}")

    for i0, data in enumerate(val_loader):
        data = data.to(device)
        input = data[:, :input_length, [1, 2, 4]]
        target = data[:, input_length:, [1, 2, 4]]

        input = scaler.transform(input)
        target = scaler.transform(target)

        for j in range(30):
            samples, log_prob = norm_model.sample(n=N*10, x=input[j].unsqueeze(0))
            _, idx = torch.topk(log_prob, k=N)
            samples = samples[:, idx[0], :, :]

            input_i = scaler.inverse_transform(input[j].unsqueeze(0))
            target_i = scaler.inverse_transform(target[j].unsqueeze(0))
            samples = scaler.inverse_transform(samples[0])

            input_i = input_i.cpu().numpy()
            target_i = target_i.cpu().numpy()
            samples = samples.cpu().numpy()

            ax = fig.add_subplot(projection='3d')
            ax.scatter(input_i[:, :, 0], input_i[:, :, 1], input_i[:, :, 2], c='r', s=1)
            ax.scatter(target_i[:, :, 0], target_i[:, :, 1], target_i[:, :, 2], c='b', s=1)

            for k in range(N):
                ax.scatter(samples[k, :, 0], samples[k, :, 1], samples[k, :, 2], c='g', s=1, alpha=5/N)

            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            # ax.set_zlim([min_z, max_z])

            ax.get_figure().savefig(f"{logdir}/fig/{epoch}/{i0}_{j}.png")
        break
