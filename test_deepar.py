from scipy.stats import multivariate_normal
from tqdm import tqdm
from util import MinMaxScaler, GaussianScaler, gauss2d
from src.Normalizing_Flow import normalizing_flow
from src.DeepAR import DeepAR
import torch
import numpy as np
import sched
import pickle
from util import AirportTrajData
import matplotlib.pyplot as plt
import matplotlib

# path = './traj.pkl'
# dataset = AirportTrajData(path, num_in=60, num_out=60)

# with open('./trajdataset.pkl', 'wb') as f:
#     pickle.dump(dataset, f)


with open('./trajdataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

np.random.seed(0)
torch.manual_seed(0)

batch_size = 512
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

norm_model = DeepAR(input_size=num_input,
                    feat_size=num_features,
                    emb_dim=emb,
                    hidden=hidden,
                    n_blocks=3,
                    num_flow_layers=10,
                    num_pred=num_pred,
                    cond_label_size=hidden,
                    flow_model="real_nvp"
                    ).to(device)

total_loss = []

norm_model.load_state_dict(torch.load('./model/epoch_2.pth'))
norm_model.eval()


# data = next(iter(test_loader))

for i0, data in enumerate(test_loader):
    data = data.to(device)
    input = data[:, :input_length, [1, 2]]
    features = data[:, :input_length, [0, 5, 6]]
    target = data[:, input_length:, [1, 2, 5, 6]]

    input = scaler.transform(input)
    target[:, :, :2] = scaler.transform(target[:, :, :2])
    features[:, :, 0] = features[:, :, 0] / (24*60*60)
    features[:, :, 1:] = gaussian_scaler.transform(features[:, :, 1:])
    info = norm_model.forward(input, target, features)

    mu = info['mu']
    log_sigma = info['log_sigma']

    mu = scaler.inverse_transform(mu)
    target = scaler.inverse_transform(target)
    input = scaler.inverse_transform(input)

    mu = mu.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    input = input.cpu().detach().numpy()

    log_sigma = log_sigma.cpu().detach().numpy()
    log_sigma = np.exp(log_sigma)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # plt.figure(figsize=(10, 10))
    # for i in range(mu.shape[0]):
    #     ax.plot(mu[i, :, 0], mu[i, :, 1], color='r', label='pred')
    #     ax.scatter(mu[i, :, 0], mu[i, :, 1], color='r', label='pred')
    #     ax.plot(target[i, :, 0], target[i, :, 1], color='b', label='target')
    #     ax.scatter(target[i, :, 0], target[i, :, 1],  color='b', label='target')
    #     ax.plot(input[i, :, 0], input[i, :, 1],  color='g', label='input')
    #     ax.scatter(input[i, :, 0], input[i, :, 1],  color='g', label='input')
    #     # plot the uncertainty
    #     ax.fill_between(mu[i, :, 0], mu[i, :, 1] - (log_sigma[i, :, 1]),
    #                     mu[i, :, 1] + (log_sigma[i, :, 1]), color='r', alpha=0.2)
    #     ax.legend()
    #     # save fig
    #     plt.savefig('./figout/{}.png'.format(i))
    #     plt.cla()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(mu.shape[0]):
        ax.plot(mu[i, :, 0], mu[i, :, 1], color='r', label='pred')
        ax.scatter(mu[i, :, 0], mu[i, :, 1], color='r', label='pred', s=0.1)
        ax.plot(target[i, :, 0], target[i, :, 1], color='b', label='target')
        ax.scatter(target[i, :, 0], target[i, :, 1],  color='b', label='target', s=0.1)
        ax.plot(input[i, :, 0], input[i, :, 1],  color='g', label='input')
        ax.scatter(input[i, :, 0], input[i, :, 1],  color='g', label='input', s=0.1)

        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        for j0 in range(6):
            j = j0 * 10
            ell = matplotlib.patches.Ellipse(xy=mu[i, j, :], width=log_sigma[i, j, 0], height=log_sigma[i, j, 1], alpha=0.1)
            ax.add_patch(ell)

        plt.savefig(f'./figout/{i0}_{i}.png')
        plt.cla()


# i = 1
# samples = norm_model.flow.sample(cond=cond[i].expand(1000, -1))
# samples = scaler.inverse_transform(samples)
# samples = samples.cpu().detach().numpy()

# input1 = input.cpu().detach().numpy()
# input1 = scaler.inverse_transform(input1)
# target1 = target.cpu().detach().numpy()
# target1 = scaler.inverse_transform(target1)

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# input_i = input1[i]
# target_i = target1[i]

# for i in range(60):
#     # ax.scatter(pred[:, :, 0], pred[:, :, 1], c='g', alpha=0.01)
#     ax.scatter(samples[:, i, 0], samples[:, i, 1], c='g')
#     # ax.plot(pred[:, :, 0], pred[:, :, 1], c='g', linewidth=1)
#     ax.scatter(input_i[:, 0], input_i[:, 1], c='b', s=10)
#     ax.plot(input_i[:, 0], input_i[:, 1], c='b', linewidth=1)
#     ax.scatter(target_i[:, 0], target_i[:, 1], c='r', s=10)
#     ax.plot(target_i[:, 0], target_i[:, 1], c='r', linewidth=1)
#     # ax.set_xlim(min(input_i[:, 0].min(), target_i[:, 0].min()), max(input_i[:, 0].max(), target_i[:, 0].max()))
#     # ax.set_ylim(min(input_i[:, 1].min(), target_i[:, 1].min()), max(input_i[:, 1].max(), target_i[:, 1].max()))

#     # save the figure
#     plt.savefig('./figout/{}.png'.format(i))
#     plt.cla()
