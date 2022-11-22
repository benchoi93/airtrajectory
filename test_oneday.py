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
from util import AirportTrajTestData, AirportTrajData
import datetime



# path = '/app/traj0530.pkl'
# dataset = AirportTrajTestData(path, num_in=60, num_out=60)

# with open('/app/trajdataset3d0530.pkl', 'wb') as f:
#     pickle.dump(dataset, f)


with open('/app/trajdataset3d0530.pkl', 'rb') as f:
    dataset:AirportTrajTestData = pickle.load(f)

with open('/app/trajdataset3d.pkl', 'rb') as f:
    traindataset:AirportTrajData = pickle.load(f)


np.random.seed(0)
torch.manual_seed(0)

# [x.shape for x in dataset.test_data]
# len(dataset.test_data)

import sys
sys.argv= ['']
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=256)

arg = parser.parse_args()

model_dir = "Flomo_2022-11-13 15:13:47.408616_absdev_input3_pred60_hidden64_emb128_alpha0.01"
model,time, encoding_type, input,pred,hidden,emb,alpha= model_dir.split("_")

batch_size = arg.batch_size
num_input = int(input.split("input")[1])
num_features = 3
num_pred = int(pred.split("pred")[1])
input_length = 60
emb = int(emb.split("emb")[1])
hidden = int(hidden.split("hidden")[1])
alpha = float(alpha.split("alpha")[1])
encoding_type = encoding_type

# num_input = 3
# num_features = 3
# num_pred = 60
# input_length = 60
# emb = 128
# hidden = 64
# alpha = 0.01
# encoding_type = "dev"
# [x.shape for x in dataset.train_data]

train_data = np.stack(traindataset.train_data, 0).astype(np.float32)
val_data = np.stack(traindataset.val_data, 0).astype(np.float32)
test_data = np.stack(dataset.test_data, 0).astype(np.float32)

# test_data[0,0]
#'tod', 'lon', 'lat', 'alt', 'fl', 'delta_lon', 'delta_lat'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_x = train_data[:, :, 1].min()
max_x = train_data[:, :, 1].max()
min_y = train_data[:, :, 2].min()
max_y = train_data[:, :, 2].max()
# min_z = train_data[:, :, 4].min()
min_z = 5
# max_z = train_data[:, :, 4].max()
max_z = 200

test_data[:, :, 4][test_data[:, :, 4] < 5] = min_z
test_data[:, :, 4][test_data[:, :, 4] > 200] = max_z
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


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
    hidden = hidden,
)

norm_model.load_state_dict(torch.load(f"/app/logs/{model_dir}/model/model_best.pt"))
norm_model.to(device)

fig = plt.figure(figsize=(20, 20))


norm_model.eval()

with torch.no_grad():
    for i, data in (pbar := tqdm(enumerate(test_loader), total=test_loader.__len__())):
        data = data.to(device)
        input = data[:, :input_length, [1, 2, 4]]
        target = data[:, input_length:, [1, 2, 4]]

        input = scaler.transform(input)
        target = scaler.transform(target)

        nllloss = norm_model.log_prob(target, input)

        samples, probs = norm_model.sample(n=1000, x=input)

        nllloss = -nllloss.mean()
        loss = nllloss