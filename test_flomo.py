from tqdm import tqdm
from util import MinMaxScaler, GaussianScaler
from src.FloMo import FloMo
import torch
import numpy as np
import sched
import pickle
import matplotlib.pyplot as plt
from util import AirportTrajData

# path = './traj.pkl'
# dataset = AirportTrajData(path, num_in=60, num_out=60)

# with open('./trajdataset.pkl', 'wb') as f:
#     pickle.dump(dataset, f)


with open('./trajdataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

np.random.seed(0)
torch.manual_seed(0)

batch_size = 128
num_input = 2
num_features = 3
num_pred = 60
input_length = 60
emb = 128
hidden = 64
alpha = 0.01


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

# define a min-max scaler
scaler = MinMaxScaler(min_x, max_x, min_y, max_y)


norm_model = FloMo(
    hist_size=input_length,
    pred_steps=num_pred,
    alpha=3,
    beta=0.002,
    gamma=0.002,
)

norm_model.load_state_dict(torch.load('./model/epoch_1.pth'))
norm_model.eval()

N = 100

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# data = next(iter(test_loader))

for i0, data in enumerate(test_loader):
    data = data.to(device)
    input = data[:, :input_length, [1, 2]]
    # features = data[:, :input_length, [0, 5, 6]]
    target = data[:, input_length:, [1, 2]]

    input = scaler.transform(input)
    target = scaler.transform(target)

    for j in range(input.shape[0]):
        samples, nll = norm_model.sample(n=N, x=input[j].unsqueeze(0))

        input_i = scaler.inverse_transform(input[j].unsqueeze(0))
        target_i = scaler.inverse_transform(target[j].unsqueeze(0))
        samples = scaler.inverse_transform(samples[0])

        input_i = input_i.cpu().numpy()
        target_i = target_i.cpu().numpy()
        samples = samples.cpu().numpy()

        ax.scatter(input_i[:, :, 0], input_i[:, :, 1], c='r', s=1)
        ax.plot(input_i[:, :, 0], input_i[:, :, 1], c='r', linewidth=0.5)
        ax.scatter(target_i[:, :, 0], target_i[:, :, 1], c='b', s=1)
        ax.plot(target_i[:, :, 0], target_i[:, :, 1], c='b', linewidth=0.5)

        for k in range(N):
            ax.scatter(samples[k, :, 0], samples[k, :, 1], c='g', s=1, alpha=0.1)
            ax.plot(samples[k, :, 0], samples[k, :, 1], c='g', linewidth=0.5, alpha=0.1)

        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        # save
        plt.savefig('./figs/{}_{}.png'.format(i0, j))

        plt.cla()
