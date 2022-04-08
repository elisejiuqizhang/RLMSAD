'''This is adapted from GitHub repo https://github.com/manigalati/usad.,
original paper was published at KDD 2020 at https://dl.acm.org/doi/10.1145/3394486.3403392, 
titled "USAD: UnSupervised Anomaly Detection on Multivariate Time Series".
Please also check the authors' original paper and implementation for reference.'''

# Set current working directory to the main branch of RLMSAD
import sys
sys.path.append('/usr/local/data/elisejzh/Projects/RLMSAD') # This is the path setting on my computer, modify this according to your need

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch

from base_detectors.USAD.usad_utils import *
from base_detectors.USAD.usad_model import *
from data_process import *
import torch.utils.data as data_utils

device = get_default_device()

path_normal="./raw_data/SWaT/SWaT_Dataset_Normal_v1.csv"
path_attack="./raw_data/SWaT/SWaT_Dataset_Attack_v0.csv"

windows_normal, _ =data_process_SWaT(path_normal,path_attack)

BATCH_SIZE =  7919
N_EPOCHS = 70
hidden_size = 40

w_size=windows_normal.shape[1]*windows_normal.shape[2]
z_size=windows_normal.shape[1]*hidden_size

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


model = UsadModel(w_size, z_size)
model = to_device(model,device)
history = training(N_EPOCHS,model,train_loader,val_loader)

filename = './base_detectors/usad_SWaT.sav'
pickle.dump(model, open(filename, 'wb'))