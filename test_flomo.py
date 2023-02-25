# from tensorboardX import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
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
from util import AirportTrajData, get_dist
import wandb

dataset = np.load("/app/data.npy")

sys.argv = [""]

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_input', type=int, default=3)
parser.add_argument('--num_features', type=int, default=3)
parser.add_argument('--num_pred', type=int, default=60)
parser.add_argument('--input_length', type=int, default=60)
parser.add_argument('--emb', type=int, default=128)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--alpha', type=float, default=9)
parser.add_argument('--beta', type=float, default=0.0002)
parser.add_argument('--gamma', type=float, default=0.00002)
parser.add_argument('--encoding_type', type=str, default="absdev", choices=["dev", "abs", "absdev"])
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)

arg = parser.parse_args()

rundir = "/app/models/230215_FloMo"

np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

batch_size = arg.batch_size
num_input = arg.num_input
num_features = arg.num_features
num_pred = arg.num_pred
input_length = arg.input_length
emb = arg.emb
hidden = arg.hidden
alpha = arg.alpha
beta = arg.beta
gamma = arg.gamma
encoding_type = arg.encoding_type
learning_rate = arg.learning_rate

data = torch.from_numpy(dataset).float()

dates = [datetime.fromtimestamp(x).date() for x in data[:, 0, 0].tolist()]
# train data : 20220523-20220527
# val data : 20220528
# test data : 20220529-20220530
train_idx = [i for i, x in enumerate(dates) if x < datetime(2022, 5, 28).date()]
val_idx = [i for i, x in enumerate(dates) if x == datetime(2022, 5, 28).date()]
test_idx = [i for i, x in enumerate(dates) if x > datetime(2022, 5, 28).date()]

train_data = data[train_idx]
val_data = data[val_idx]
test_data = data[test_idx]

train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=True, pin_memory=True)

# gaussian_scaler = GaussianScaler(train_data[:, :, -2:])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


min_x = train_data[:, :, 1].min()
max_x = train_data[:, :, 1].max()
min_y = train_data[:, :, 2].min()
max_y = train_data[:, :, 2].max()
min_z = train_data[:, :, 3].min()
max_z = train_data[:, :, 3].max()
# min_z = 5
# max_z = 200

# train_data[:, :, 4][train_data[:, :, 4] < 5] = min_z
# train_data[:, :, 4][train_data[:, :, 4] > 200] = max_z

# define a min-max scaler
scaler = MinMaxScaler(min_x, max_x, min_y, max_y, min_z, max_z)

norm_model = FloMo(
    hist_size=input_length,
    pred_steps=num_pred,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    num_in=num_input,
    encoding_type=encoding_type,
    hidden=hidden,
)

total_loss = []

fig = plt.figure(figsize=(20, 20))

norm_model.load_state_dict(torch.load(os.path.join(rundir, f"model_best.pt")))
norm_model.eval()
performance = np.zeros((len(test_loader) * batch_size, 12))
cnt = 0

with torch.no_grad():
    for i, (data,) in (pbar := tqdm(enumerate(test_loader), total=test_loader.__len__())):
        data = data.to(device)
        input = data[:, :input_length, [1, 2, 3]]
        features = data[:, :input_length, [4, 5, 6]]
        target = data[:, input_length:, [1, 2, 3]]

        input = scaler.transform(input)
        target = scaler.transform(target)
        input = torch.concat((input, features), 2)

        samples, probs = norm_model.sample(n=100, x=input)
        samples = scaler.inverse_transform(samples)
        samples[samples < 0] = 0
        target = scaler.inverse_transform(target)
        input_abs = scaler.inverse_transform(input[:, :, :3])

        dist = get_dist(samples, target)

        performance[(cnt):(cnt+samples.shape[0]), 0] = dist.mean(-1).min(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 1] = dist.mean(-1).max(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 2] = dist.mean(-1).mean(-1).cpu().numpy()

        # calculate minFDE
        performance[(cnt):(cnt+samples.shape[0]), 3] = dist[..., -1].min(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 4] = dist[..., -1].max(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 5] = dist[..., -1].mean(-1).cpu().numpy()
        cnt += samples.shape[0]

        samples = samples.cpu().numpy()
        target = target.cpu().numpy()
        input_abs = input_abs.cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for j in range(samples.shape[0]):
            # draw 3d plot
            for k in range(samples.shape[1]):
                ax.scatter(samples[j, k, :, 0], samples[j, k, :, 1], samples[j, k, :, 2], c='green', alpha=0.01)
            ax.scatter(target[j, :, 0], target[j, :, 1], target[j, :, 2], c='red')
            ax.scatter(input_abs[j, :, 0], input_abs[j, :, 1], input_abs[j, :, 2], c='blue')
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

            # save
            plt.savefig(os.path.join(rundir, f"test_{i}_{j}.png"))

            # clear
            ax.clear()


minADE = performance[:, 0].mean()
maxADE = performance[:, 1].mean()
avgADE = performance[:, 2].mean()
minFDE = performance[:, 3].mean()
maxFDE = performance[:, 4].mean()
avgFDE = performance[:, 5].mean()

print(
    f"minADE: {minADE:.2f}, maxADE: {maxADE:.2f}, avgADE: {avgADE:.2f}, minFDE: {minFDE:.2f}, maxFDE: {maxFDE:.2f}, avgFDE: {avgFDE:.2f}"
)
