'''This is adapted from GitHub repo https://github.com/manigalati/usad.,
original paper was published at KDD 2020 at https://dl.acm.org/doi/10.1145/3394486.3403392, 
titled "USAD: UnSupervised Anomaly Detection on Multivariate Time Series".
Please also check the authors' original paper and implementation for reference.'''

# Set current working directory to the main branch of RLMSAD
import sys
sys.path.append('/usr/local/data/elisejzh/Projects/RLMSAD') # This is the path setting on my computer, modify this according to your need

import numpy as np
import pandas as pd
from sklearn import preprocessing


def data_process_SWaT(path_normal,path_attack,down_rate=5,window_size=12):  
    '''Data Processing Function Specifically for the SWaT Dataset:
        Inputs: 
            path_normal: file path to the SWaT normal data in csv;
            path_attack: file path to the SWaT attack data in csv;
            down_rate: downsampling rate of the original data (by taking the mean), reduce the complexity;
            window_size: the size of the sliding window to be feed to the base models.
        Outputs:
            windows_normal: sliding windows cropped from the normal data, to be feed to the base models;
            windows_attack: sliding windows cropped from the attack data, to be feed to the base models.'''
    
    # Handling Normal Data
    #Read
    normal = pd.read_csv(path_normal)
    normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
    #Downsampling
    normal=normal.groupby(np.arange(len(normal.index)) // down_rate).mean()
    #Transform all columns into float64
    for i in list(normal): 
        normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
    normal = normal.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = normal.values
    x_scaled = min_max_scaler.fit_transform(x)
    normal = pd.DataFrame(x_scaled)

    # Handling Attack Data
    #Read
    attack = pd.read_csv(path_attack,sep=";")
    labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
    attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
    #Downsampling the attack data
    attack=attack.groupby(np.arange(len(attack.index)) // down_rate).mean()
    #Downsampling the labels
    labels_down=[]
    for i in range(len(labels)//down_rate):
        if labels[5*i:5*(i+1)].count(1.0):
            labels_down.append(1.0) #Attack labels
        else:
            labels_down.append(0.0) #Normal labels
    #for the last few labels that are not included in a full-length window
    if labels[down_rate*(i+1):].count(1.0):
        labels_down.append(1.0) #Attack labels
    else:
        labels_down.append(0.0) #Normal labels
    # Transform all columns into float64
    for i in list(attack):
        attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
    attack = attack.astype(float)
    #Normalization
    x = attack.values 
    x_scaled = min_max_scaler.transform(x)
    attack = pd.DataFrame(x_scaled)

    # Create Sliding Windows
    windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
    windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]

    return windows_normal, windows_attack, labels_down