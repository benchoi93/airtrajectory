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
import datetime
import pandas as pd


with open('/app/trajdataset3d.pkl', 'rb') as f:
    dataset = pickle.load(f)

np.random.seed(0)
torch.manual_seed(0)

logdir = "Flomo_2022-11-12 21:04:03.107773_absdev_input3_pred60_hidden64_emb128_alpha0.01"
# logdir = "Flomo_2022-11-12 21:04:03.140917_abs_input3_pred60_hidden64_emb128_alpha0.01"
# logdir = "Flomo_2022-11-12 21:04:05.144979_dev_input3_pred60_hidden64_emb128_alpha0.01"

def test(logdir):
    model, logtime, encoding_type , input, pred, hidden, emb, alpha= logdir.split("_")

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

    norm_model.load_state_dict(torch.load(os.path.join("/app/logs", logdir, "model","model_30.pt")))
    norm_model.eval()

    performance = np.zeros((len(test_loader)* batch_size, 12))
    cnt =0
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

            samples = scaler.inverse_transform(samples)
            target = scaler.inverse_transform(target)

            # calculate minADE
            performance[(cnt):(cnt+samples.shape[0]),0] =  (samples - target.unsqueeze(1)).pow(2).sum(-1).sqrt().mean(-1).min(-1)[0].cpu().numpy()
            performance[(cnt):(cnt+samples.shape[0]),1] =  (samples - target.unsqueeze(1)).pow(2).sum(-1).sqrt().mean(-1).max(-1)[0].cpu().numpy()
            performance[(cnt):(cnt+samples.shape[0]),2] =  (samples - target.unsqueeze(1)).pow(2).sum(-1).sqrt().mean(-1).mean(-1).cpu().numpy()

            # calculate minFDE
            performance[(cnt):(cnt+samples.shape[0]),3] =  (samples[:,:,-1,:] - target[:,-1,:].unsqueeze(1)).pow(2).sum(-1).sqrt().min(-1)[0].cpu().numpy()
            performance[(cnt):(cnt+samples.shape[0]),4] =  (samples[:,:, -1, :] - target[:,-1,:].unsqueeze(1)).pow(2).sum(-1).sqrt().max(-1)[0].cpu().numpy()
            performance[(cnt):(cnt+samples.shape[0]),5] =  (samples[:,:, -1, :] - target[:,-1,:].unsqueeze(1)).pow(2).sum(-1).sqrt().mean(-1).cpu().numpy()

            # calculate top K ADE
            _, idx = torch.topk(probs, k=100)
            samples = samples[:, idx[0], :, :]

            # calculate minADE
            performance[(cnt):(cnt+samples.shape[0]),6] =  (samples - target.unsqueeze(1)).pow(2).sum(-1).sqrt().mean(-1).min(-1)[0].cpu().numpy()
            performance[(cnt):(cnt+samples.shape[0]),7] =  (samples - target.unsqueeze(1)).pow(2).sum(-1).sqrt().mean(-1).max(-1)[0].cpu().numpy()
            performance[(cnt):(cnt+samples.shape[0]),8] =  (samples - target.unsqueeze(1)).pow(2).sum(-1).sqrt().mean(-1).mean(-1).cpu().numpy()

            # calculate minFDE
            performance[(cnt):(cnt+samples.shape[0]),9] =  (samples[:,:,-1,:] - target[:,-1,:].unsqueeze(1)).pow(2).sum(-1).sqrt().min(-1)[0].cpu().numpy()
            performance[(cnt):(cnt+samples.shape[0]),10] =  (samples[:,:, -1, :] - target[:,-1,:].unsqueeze(1)).pow(2).sum(-1).sqrt().max(-1)[0].cpu().numpy()
            performance[(cnt):(cnt+samples.shape[0]),11] =  (samples[:,:, -1, :] - target[:,-1,:].unsqueeze(1)).pow(2).sum(-1).sqrt().mean(-1).cpu().numpy()

            cnt += samples.shape[0]        


    print(f"minADE: {performance[:, 0].mean():.4f}")
    print(f"maxADE: {performance[:, 1].mean():.4f}")
    print(f"meanADE: {performance[:, 2].mean():.4f}")

    print(f"minFDE: {performance[:, 3].mean():.4f}")
    print(f"maxFDE: {performance[:, 4].mean():.4f}")
    print(f"meanFDE: {performance[:, 5].mean():.4f}")

    print(f"TopK minADE: {performance[:, 6].mean():.4f}")
    print(f"TopK maxADE: {performance[:, 7].mean():.4f}")
    print(f"TopK meanADE: {performance[:, 8].mean():.4f}")

    print(f"TopK minFDE: {performance[:, 9].mean():.4f}")
    print(f"TopK maxFDE: {performance[:, 10].mean():.4f}")
    print(f"TopK meanFDE: {performance[:, 11].mean():.4f}")

    return performance.mean(0)

if __name__ == "__main__":

    logdir = "/app/logs"

    logdir_list = os.listdir(logdir)
    result = np.zeros((len(logdir_list), 16))

    i=0
    for logdir in logdir_list:
        print(logdir)
        performance = test(logdir)

        
        _, _, encoding_type , _, _, hidden, emb, alpha= logdir.split("_")
        emb = int(emb.split("emb")[1])
        hidden = int(hidden.split("hidden")[1])
        alpha = float(alpha.split("alpha")[1])
        encoding_type = encoding_type

        result[i,0] = emb
        result[i,1] = hidden
        result[i,2] = alpha
        result[i,3] = encoding_type
        
        result[i,4:] = performance

        pd.DataFrame(result).to_csv("result.csv")
        i+=1