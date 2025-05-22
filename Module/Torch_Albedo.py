# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Torch_Albedo.py
@Time: 2023/10/8 15:27
@Function: 
"""

import os
# import seaborn as sns
import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


# create ds class
class MyDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):           # return ts
        x, y = self.x[index], self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)


# train ---
# RMSE
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class NN(nn.Module):

    def __init__(self, n_features, n_hidden):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def predict_albedo(df_test):

    # print(df_test)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # df_norm_albedo = pd.read_csv(r"/home/ritascake/wangq/proj_monsoon/DL_Albedo/z_norm.csv", index_col=0).drop('albedo')
    df_norm_albedo = pd.read_csv(r"/home/ritascake/wangq/proj_dl/modis_albedo/DL-Train/z_norm.csv", index_col=0).drop('albedo')

    # if 'debris' in df_test.columns:
    #     df_test.drop('debris', axis=1, inplace=True)

    df_test = (df_test - df_norm_albedo['min']) / (df_norm_albedo['max'] - df_norm_albedo['min']) * 100
    # model = torch.load(r"/home/ritascake/wangq/proj_monsoon/DL_Albedo/model/ann_albedo_model_SETP_147.pt", map_location=device)
    model = torch.load(r"/home/ritascake/wangq/proj_dl/modis_albedo/DL-Train/ann_albedo_model.pt", map_location=device)

    test = torch.FloatTensor(df_test.to_numpy())
    test = test.to(device)

    pred_nn = model(test).cpu().detach().numpy()

    # pred_nn = pred_nn / 100 * (df_norm_albedo['max'] - df_norm_albedo['min']) + df_norm_albedo['min']

    return pred_nn
