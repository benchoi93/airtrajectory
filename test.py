from tqdm import tqdm
from util import MinMaxScaler, GaussianScaler
from src.Normalizing_Flow import normalizing_flow
import torch
import numpy as np
import sched
import pickle
from util import AirportTrajData
import matplotlib.pyplot as plt
# path = './traj.pkl'
# dataset = AirportTrajData(path, num_in=60, num_out=60)

# with open('./trajdataset.pkl', 'wb') as f:
#     pickle.dump(dataset, f)


with open('./trajdataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

np.random.seed(0)
torch.manual_seed(0)

batch_size = 1024
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

min_x = 126.52400
max_x = 126.99300
min_y = 37.34100
max_y = 37.71800


# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.scatter(test_data[:,:,1], test_data[:,:,2], s=0.1)
# # draw a rectangle
# rect = plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# plt.show()


# define a min-max scaler
scaler = MinMaxScaler(min_x, max_x, min_y, max_y)

norm_model = normalizing_flow(input_size=num_input,
                              feat_size=num_features,
                              emb_dim=emb,
                              hidden=hidden,
                              n_blocks=3,
                              num_flow_layers=10,
                              num_pred=num_pred,
                              cond_label_size=hidden,
                              flow_model="real_nvp"
                              ).to(device)

optim = torch.optim.Adam(norm_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
total_loss = []

norm_model.load_state_dict(torch.load('./model/epoch_3.pth'))
norm_model.eval()

data = next(iter(train_loader))
data = data.to(device)
input = data[:, :input_length, [1, 2]]
features = data[:, :input_length, [0, 5, 6]]
target = data[:, input_length:, [1, 2]]

input = scaler.transform(input)
target = scaler.transform(target)
features[:, :, 0] = features[:, :, 0] / (24*60*60)
features[:, :, 1:] = gaussian_scaler.transform(features[:, :, 1:])

# add small noise to target
target = target + torch.randn_like(target) * 0.0005

cond = norm_model.gru_model(input, features)

i = 1
samples = norm_model.flow.sample(cond=cond[i].expand(1000, -1))
samples = scaler.inverse_transform(samples)
samples = samples.cpu().detach().numpy()

input1 = input.cpu().detach().numpy()
input1 = scaler.inverse_transform(input1)
target1 = target.cpu().detach().numpy()
target1 = scaler.inverse_transform(target1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

input_i = input1[i]
target_i = target1[i]

for i in range(60):
    # ax.scatter(pred[:, :, 0], pred[:, :, 1], c='g', alpha=0.01)
    ax.scatter(samples[:, i, 0], samples[:, i, 1], c='g')
    # ax.plot(pred[:, :, 0], pred[:, :, 1], c='g', linewidth=1)
    ax.scatter(input_i[:, 0], input_i[:, 1], c='b', s=10)
    ax.plot(input_i[:, 0], input_i[:, 1], c='b', linewidth=1)
    ax.scatter(target_i[:, 0], target_i[:, 1], c='r', s=10)
    ax.plot(target_i[:, 0], target_i[:, 1], c='r', linewidth=1)
    # ax.set_xlim(min(input_i[:, 0].min(), target_i[:, 0].min()), max(input_i[:, 0].max(), target_i[:, 0].max()))
    # ax.set_ylim(min(input_i[:, 1].min(), target_i[:, 1].min()), max(input_i[:, 1].max(), target_i[:, 1].max()))

    # save the figure
    plt.savefig('./figout/{}.png'.format(i))
    plt.cla()
