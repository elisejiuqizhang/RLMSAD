'''Evaluate each base model and the basic ensemble (majority vote)'''

# Set current working directory to the main branch of RLMSAD
import os
import sys
sys.path.append('/usr/local/data/elisejzh/Projects/RLMSAD') # This is the path setting on my computer, modify this according to your need

from data_process import *
from base_detectors.get_preds_thres import *

from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.utils.data as data_utils
import pickle

path_normal="./raw_data/SWaT/SWaT_Dataset_Normal_v1.csv"
path_attack="./raw_data/SWaT/SWaT_Dataset_Attack_v0.csv"

# Prepare for computing raw scores and get ground truth labels
windows_normal, windows_attack, labels_down =data_process_SWaT(path_normal,path_attack)

BATCH_SIZE =  7919
N_EPOCHS = 70
hidden_size = 40

w_size=windows_normal.shape[1]*windows_normal.shape[2]

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
                    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))), 
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load the pretrained model
model_path='./base_detectors/iforest_SWaT.sav'
loaded_model = pickle.load(open(model_path, 'rb'))

# Predicted scores
windows_attack=windows_attack.reshape((windows_attack.shape[0],-1)) # Flatten the windows
sc_pred=-loaded_model.decision_function(windows_attack)# sklearn defined this as the opposite as the original paper, so here it is the lower the more abnormal
print("Max score:",np.max(sc_pred))
print("Min score:",np.min(sc_pred))


# Ground truth labels
windows_labels=[]
for i in range(len(labels_down)-windows_normal.shape[1]):
    windows_labels.append(list(np.int_(labels_down[i:i+windows_normal.shape[1]])))
list_gtruth=[1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

# default threshold
default_thres=raw_thresholds(sc_pred,contamination=0.11)
print("Default threshold:",default_thres)

# Try different thresholds
thres=raw_thresholds(sc_pred,contamination=0.12)
# thres=0.1
y_pred_label = [1.0 if (score > thres) else 0 for score in sc_pred ]

prec=precision_score(list_gtruth,y_pred_label,pos_label=1)
rec=recall_score(list_gtruth,y_pred_label,pos_label=1)
f1=f1_score(list_gtruth,y_pred_label,pos_label=1)

print("Precision:",prec)
print("Recall:",rec)
print("F1:",f1)