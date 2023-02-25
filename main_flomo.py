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

# path = '/app/traj.pkl'
# dataset = AirportTrajData(path, num_in=60, num_out=60)

# with open('/app/trajdataset3d.pkl', 'wb') as f:
#     pickle.dump(dataset, f)


# with open('/app/trajdataset3d.pkl', 'rb') as f:
#     dataset = pickle.load(f)

dataset = np.load("/app/data.npy")

# sys.argv = [""]

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

wandb.init(
    project="AirNew",
    entity="benchoi93",
    config=arg,
    # settings=wandb.Settings(start_method="fork")
)


# num_input = 3
# num_features = 3
# num_pred = 60
# input_length = 60
# emb = 128
# hidden = 64
# alpha = 0.01
# encoding_type = "dev"
# [x.shape for x in dataset.train_data]

# train_data = np.stack(dataset.train_data, 0).astype(np.float32)
# val_data = np.stack(dataset.val_data, 0).astype(np.float32)
# test_data = np.stack(dataset.test_data, 0).astype(np.float32)

# dataset['X'].shape
# dataset["Y"].shape


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
test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False, pin_memory=True)

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


optim = torch.optim.Adam(norm_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
total_loss = []

fig = plt.figure(figsize=(20, 20))

# logdir = f"./logs/Flomo_{datetime.datetime.now()}_{encoding_type}_input{num_input}_pred{num_pred}_hidden{hidden}_emb{emb}_alpha{alpha}_beta{beta}_gamma{gamma}"
# writer = SummaryWriter(logdir)
# create directory "./model" for saving model
# if not os.path.exists(f"{logdir}/model"):
#     os.makedirs(f"{logdir}/model")
# if not os.path.exists(f"{logdir}/fig"):
#     os.makedirs(f"{logdir}/fig")

for epoch in range(arg.epoch):
    norm_model.train()

    losses = 0
    sample_mae_sum = 0
    sample_mae_sum_x = 0
    sample_mae_sum_y = 0
    sample_mae_sum_z = 0
    cnt = 0
    for i, (data,) in (pbar := tqdm(enumerate(train_loader), total=train_loader.__len__())):
        optim.zero_grad()
        data = data.to(device)
        input = data[:, :input_length, [1, 2, 3]]
        features = data[:, :input_length, [4, 5, 6]]
        target = data[:, input_length:, [1, 2, 3]]

        input = scaler.transform(input)
        target = scaler.transform(target)
        input = torch.concat((input, features), 2)
        # features[:, :, 0] = features[:, :, 0] / (24*60*60)
        # features[:, :, 1:] = gaussian_scaler.transform(features[:, :, 1:])

        # add small noise to target
        # target = [target + torch.randn_like(target) * 0.0005 for _ in range(5)]
        # target = torch.concat(target, 0)
        # input = input.repeat(5, 1, 1)
        # features = features.repeat(5, 1, 1)

        nllloss = norm_model.log_prob(target, input)

        nllloss = -nllloss.mean()
        # loss = alpha * nllloss + (1-alpha) * mae_loss
        loss = nllloss

        loss.backward()
        optim.step()
        losses += loss.item()

        cnt += 1
        pbar.set_description(
            f"Train Epoch {epoch} | Loss {loss.item():.4f} ")
        # pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f} | NLLLoss {nllloss.item():.4f} | MAELoss {mae_loss.item()/N:.4f}")

        if i == train_loader.__len__()-1:
            with torch.no_grad():
                samples, probs = norm_model.sample(n=100, x=input)
                samples = scaler.inverse_transform(samples)
                target = scaler.inverse_transform(target)

                dist = get_dist(samples, target)

            total_loss.append(losses/cnt)
            wandb.log({
                "train/loss": losses/cnt,
                "train/ADE_min": dist.mean(-1).min(-1)[0].mean().item(),
                "train/ADE_avg": dist.mean(-1).mean(-1).mean().item(),
                "train/ADE_max": dist.mean(-1).max(-1)[0].mean().item(),
                "train/FDE_min": dist[:, -1, :].min(-1)[0].mean().item(),
                "train/FDE_avg": dist[:, -1, :].mean(-1).mean().item(),
                "train/FDE_max": dist[:, -1, :].max(-1)[0].mean().item(),
            }, step=epoch)

            pbar.set_description(f"Train Epoch {epoch} | Loss {losses/cnt} ")

    scheduler.step()
    # save
    torch.save(norm_model.state_dict(), os.path.join(wandb.run.dir, f"model_{epoch}.pt"))
    # wandb.save(os.path.join(wandb.run.dir, f"model_{epoch}.pt"))

    norm_model.eval()
    N = 30

    losses = 0
    sample_mae_sum = 0
    sample_mae_sum_x = 0
    sample_mae_sum_y = 0
    sample_mae_sum_z = 0
    cnt = 0

    best_performance = 100000

    with torch.no_grad():
        for i, (data,) in (pbar := tqdm(enumerate(val_loader), total=val_loader.__len__())):
            data = data.to(device)
            input = data[:, :input_length, [1, 2, 3]]
            features = data[:, :input_length, [4, 5, 6]]
            target = data[:, input_length:, [1, 2, 3]]

            input = scaler.transform(input)
            target = scaler.transform(target)
            input = torch.concat((input, features), 2)

            nllloss = norm_model.log_prob(target, input)

            nllloss = -nllloss.mean()
            loss = nllloss

            losses += loss.item()

            cnt += 1
            pbar.set_description(
                f"Val Epoch {epoch} | Loss {loss.item():.4f} ")

            if i == val_loader.__len__()-1:
                samples, probs = norm_model.sample(n=100, x=input)
                samples = scaler.inverse_transform(samples)
                target = scaler.inverse_transform(target)
                dist = get_dist(samples, target)

                total_loss.append(losses/cnt)
                wandb.log({
                    "val/loss": losses/cnt,
                    "val/minADE": dist.mean(-1).min(-1)[0].mean().item(),
                    "val/avgADE": dist.mean(-1).mean(-1).mean().item(),
                    "val/maxADE": dist.mean(-1).max(-1)[0].mean().item(),
                    "val/minFDE": dist[:, -1, :].min(-1)[0].mean().item(),
                    "val/avgFDE": dist[:, -1, :].mean(-1).mean().item(),
                    "val/maxFDE": dist[:, -1, :].max(-1)[0].mean().item(),
                }, step=epoch)

                pbar.set_description(f"Val Epoch {epoch} | Loss {loss.item():.4f} ")

                if best_performance > losses/cnt:
                    best_performance = losses/cnt
                    torch.save(norm_model.state_dict(), os.path.join(wandb.run.dir, f"model_best.pt"))
                    wandb.save(os.path.join(wandb.run.dir, f"model_best.pt"))


# test
norm_model.load_state_dict(torch.load(os.path.join(wandb.run.dir, f"model_best.pt")))
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
        target = scaler.inverse_transform(target)

        dist = get_dist(samples, target)

        performance[(cnt):(cnt+samples.shape[0]), 0] = dist.mean(-1).min(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 1] = dist.mean(-1).max(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 2] = dist.mean(-1).mean(-1).cpu().numpy()

        # calculate minFDE
        performance[(cnt):(cnt+samples.shape[0]), 3] = dist[..., -1].min(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 4] = dist[..., -1].max(-1)[0].cpu().numpy()
        performance[(cnt):(cnt+samples.shape[0]), 5] = dist[..., -1].mean(-1).cpu().numpy()
        cnt += samples.shape[0]



minADE = performance[:, 0].mean()
maxADE = performance[:, 1].mean()
avgADE = performance[:, 2].mean()
minFDE = performance[:, 3].mean()
maxFDE = performance[:, 4].mean()
avgFDE = performance[:, 5].mean()

wandb.log({
    "test/minADE": minADE,
    "test/avgADE": avgADE,
    "test/maxADE": maxADE,
    "test/minFDE": minFDE,
    "test/avgFDE": avgFDE,
    "test/maxFDE": maxFDE,
})
