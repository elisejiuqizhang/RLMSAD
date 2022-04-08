# Set current working directory to the main branch of RLMSAD
import os
import sys
sys.path.append('/usr/local/data/elisejzh/Projects/RLMSAD') # This is the path setting on my computer, modify this according to your need
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from base_detectors.get_preds_thres import *

# Make predictions based on the pretrained models, obtain the raw scores and thresholds
list_pred_sc, list_gtruth=raw_scores_gtruth(path_normal="./raw_data/SWaT/SWaT_Dataset_Normal_v1.csv",
                                            path_attack="./raw_data/SWaT/SWaT_Dataset_Attack_v0.csv")
# thresholds - FULL dataset
list_thresholds=[]
for i in range(len(list_pred_sc)):
    list_thresholds.append(raw_thresholds(list_pred_sc[i],contamination=0.12))

# Predicted labels for each model
list_pred_labels=[]
for i in range(len(list_pred_sc)): # For each model
    pred_tmp=np.zeros(len(list_pred_sc[i]))
    for j in range(len(list_pred_sc[i])): # For each sample
        if list_pred_sc[i][j]>=list_thresholds[i]: # if the score is larger than the threshold, the label is 1 (anomaly)
            pred_tmp[j]=1
    list_pred_labels.append(pred_tmp)

# Obtain the majority vote labels
list_pred_labels_maj=[]
for j in range(len(list_pred_sc[i])): # for each sample
    num_a_tmp=0 # number of models have predicted 1 (anomaly)
    for i in range(len(list_pred_sc)): # for each model
        if list_pred_labels[i][j]==1: # if the label is 1 (anomaly)
            num_a_tmp+=1
    if num_a_tmp>=len(list_pred_sc)/2: # if the number of models have predicted 1 (anomaly) is larger than half of the models, the label is 1 (anomaly)
        list_pred_labels_maj.append(1)
    else:
        list_pred_labels_maj.append(0)

# Evaluate the model
prec=precision_score(list_gtruth, list_pred_labels_maj)
rec=recall_score(list_gtruth, list_pred_labels_maj)
f1=f1_score(list_gtruth, list_pred_labels_maj)

print("Total number of reported anomalies: ",sum(list_pred_labels_maj))
print("Total number of true anomalies: ",sum(list_gtruth))

print("Precision: %.4f" % (prec))
print("Recall: %.4f" % (rec))
print("F1: %.4f" % (f1))