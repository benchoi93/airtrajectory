from tqdm import tqdm
from util import MinMaxScaler, GaussianScaler
from src.Normalizing_Flow import normalizing_flow
import torch
import numpy as np
import sched
import pickle
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

for epoch in range(100):
    norm_model.train()

    losses = 0
    cnt = 0
    for i, data in (pbar := tqdm(enumerate(train_loader), total=train_loader.__len__())):
        optim.zero_grad()
        data = data.to(device)
        input = data[:, :input_length, [1, 2]]
        features = data[:, :input_length, [0, 5, 6]]
        target = data[:, input_length:, [1, 2]]

        input = scaler.transform(input)
        target = scaler.transform(target)
        features[:, :, 0] = features[:, :, 0] / (24*60*60)
        features[:, :, 1:] = gaussian_scaler.transform(features[:, :, 1:])

        # add small noise to target
        target = [target + torch.randn_like(target) * 0.0005 for _ in range(5)]
        target = torch.concat(target, 0)

        input = input.repeat(5, 1, 1)
        features = features.repeat(5, 1, 1)

        nllloss = norm_model.forward(input,  target, features)

        nllloss = nllloss.mean()
        # loss = alpha * nllloss + (1-alpha) * mae_loss
        loss = nllloss

        loss.backward()
        optim.step()
        losses += loss.item()
        cnt += 1
        pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f} ")
        # pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f} | NLLLoss {nllloss.item():.4f} | MAELoss {mae_loss.item()/N:.4f}")

        if i == train_loader.__len__()-1:
            total_loss.append(losses/cnt)
            pbar.set_description(f"Epoch {epoch} | Loss {losses/cnt:.4f}")

    scheduler.step()
    # save
    torch.save(norm_model.state_dict(), f'./model/epoch_{epoch}.pth')
