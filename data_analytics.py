''' 
DATA FIELD
---------------------------------------------------
date = UTC datetime
area = RKSS:김포공항, RKPC:인천공항
source = ADS-B
tid = Callsign
dep = 출발 공항
arr = 도착 공항

dof = day of flight
tod = time of day (0~86400)

lat = latitude
lon = longitude
alt = altitude
fl = flight level
ssr = Secondary Surveillance Radar (sqarwk) code
gufi = Global Unique Flight Identifier
---------------------------------------------------
'''
import pickle
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
os.chdir("/app")
data_folder_original = os.listdir("data")

for i0 in range(len(data_folder_original)):
    data_folder = [data_folder_original[i0]]
    # read all csvs in data_folder and concat
    df = pd.concat([pd.read_csv("data/"+f, header=None) for f in data_folder], ignore_index=True)
    df.columns = ['date', 'area', 'source', 'tid', 'dep', 'arr', 'dof', 'tod', 'lat', 'lon', 'alt', 'fl', 'ssr', 'gufi']

    df = df[df['date'] != 'date']
    df.lon = df.lon.astype(float)
    df.lat = df.lat.astype(float)

    print(df.shape)
    # df lon should be between 37 and 38
    # df lat should be between 126 and 127
    df = df[(df['lon'] >= 126) & (df['lon'] <= 127)]
    df = df[(df['lat'] >= 37) & (df['lat'] <= 38)]

    print(df.shape)

    plt.plot(df.lon, df.lat, 'o', markersize=0.1)


    # len(df['tid'].unique()) # 11917

    # df_RKSS = df[df['area'] == 'rkss']
    # df_RKPC = df[df['area'] == 'rkpc']
    # print(len(df_RKSS['tid'].unique())) # 11917
    # print(len(df_RKPC['tid'].unique())) # 0

    result = []
    obs_len = 120
    for tid in (pbar := tqdm(df['tid'].unique())):
        df_tid = df[df['tid'] == tid]
        df_tid = df_tid.sort_values(by=['date'])
        df_tid = df_tid.reset_index(drop=True)

        new_tid = []
        # time_diff_list = []
        prev_t = datetime.datetime.strptime(df_tid.iloc[0]['date'], '%Y-%m-%d %H:%M:%S %Z')
        cnt = 1

        df_tid2 = df_tid.copy()
        df_tid2 = df_tid2[['date', 'lat', 'lon', 'alt', 'fl']]
        df_tid2['tod'] = df_tid2['date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S %Z') -
                                            datetime.datetime.strptime(x.split(' ')[0], '%Y-%m-%d')).total_seconds())
        df_tid2.tod = df_tid2.tod.astype(int)

        # sort by date
        df_tid2 = df_tid2.sort_values(by=['date'])

        # cut trajectory if the time difference between two consecutive rows is greater than 10 minutes
        # and apeend to result

        for i in range(len(df_tid2)):
            t = datetime.datetime.strptime(df_tid2.iloc[i]['date'], '%Y-%m-%d %H:%M:%S %Z')
            time_diff = (t - prev_t).total_seconds()
            # print(time_diff)
            if time_diff > 600:
                if cnt >= obs_len:
                    result.append(df_tid2.iloc[(i-cnt+1):(i)])
                cnt = 1
            else:
                cnt += 1
            prev_t = t

        pbar.set_description(f"tid: {tid}, len: {len(result)}")


    with open(f'result{data_folder_original[i0].split("-")[-2]}{data_folder_original[i0].split("-")[-1].split(".csv")[0]}.pkl', 'wb') as f:
        pickle.dump(result, f)






    # result2 = [x for x in result if len(x) > 120]

    # print(len(result))
    # print(len(result2))

    # (datetime.datetime.strptime(result2[0][-1][0], '%Y-%m-%d %H:%M:%S %Z') -
    #  datetime.datetime.strptime(result2[0][0][0], '%Y-%m-%d %H:%M:%S %Z')).total_seconds()


    # len(result2[0])

    # result2[0][-1]


    # num_in = 20
    # min_len = 50
    # for traj in result:
    #     if len(traj) > min_len:
    #         for i in range(len(traj)-num_in):
    #             print(traj[i:i+num_in+1])


    # fig, ax = plt.subplots(figsize=(20, 20))
    # print(len(result))
    # result = [x for x in result if len(x) > 50]
    # print(len(result))
    # cnt = 0
    # for traj in tqdm(result):
    #     # draw traj and save
    #     plt.plot([x[2] for x in traj], [x[1] for x in traj])
    #     plt.xlim([126.4, 127.1])
    #     plt.ylim([37.2, 37.9])
    #     # set fig size
    #     plt.savefig(f"fig/traj_{cnt}.png")
    #     plt.clf()
    #     cnt += 1

    # np.histogram([len(x) for x in result], bins=100)
