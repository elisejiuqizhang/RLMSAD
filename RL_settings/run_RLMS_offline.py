# Set current working directory to the main branch of RLMSAD
import os
import sys

from numpy import average
sys.path.append('/usr/local/data/elisejzh/Projects/RLMSAD') # This is the path setting on my computer, modify this according to your need
from RL_settings.env import *
from base_detectors.get_preds_thres import *

from stable_baselines3 import DQN,PPO,A2C

# Make predictions based on the pretrained models, obtain the raw scores and thresholds
list_pred_sc, list_gtruth=raw_scores_gtruth(path_normal="./raw_data/SWaT/SWaT_Dataset_Normal_v1.csv",
                                            path_attack="./raw_data/SWaT/SWaT_Dataset_Attack_v0.csv")
# thresholds
list_thresholds=[]
for i in range(len(list_pred_sc)):
    list_thresholds.append(raw_thresholds(list_pred_sc[i],contamination=0.12))



EXP_TIMES=10 # How many runs to average the results

# Store the precision, recall, F1-score
store_prec=np.zeros(EXP_TIMES)
store_rec=np.zeros(EXP_TIMES)
store_f1=np.zeros(EXP_TIMES)

for times in range(EXP_TIMES):
    # Set up the training environment on all the dataset
    # env_off=TrainEnvOffline(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)
    # env_off=TrainEnvOffline_consensus_conf(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)
    env_off=TrainEnvOffline_dist_conf(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)

    # Train the model on all the dataset  
    model = DQN('MlpPolicy', env_off, verbose=0)
    # model=A2C('MlpPolicy', env_off, verbose=0)
    model.learn(total_timesteps=len(list_pred_sc[0]))
    # model.save("DQN_offline_model")
    # model.save("A2C_offline_model")
    
    # Evaluate the model on all the dataset
    # model = DQN.load("DQN_offline_model")
    # model=A2C.load("A2C_offline_model")
    prec, rec, f1, _, list_preds=eval_model(model, env_off)

    store_prec[times]=prec
    store_rec[times]=rec
    store_f1[times]=f1

# Compute the mean and standard deviation of the results
average_prec=np.mean(store_prec)
average_rec=np.mean(store_rec)
average_f1=np.mean(store_f1)

std_prec=np.std(store_prec)
std_rec=np.std(store_rec)
std_f1=np.std(store_f1)

print("Total number of reported anomalies: ",sum(list_preds))
print("Total number of true anomalies: ",sum(list_gtruth))

print("Average precision: %.4f, std: %.4f" % (average_prec, std_prec))
print("Average recall: %.4f, std: %.4f" % (average_rec, std_rec))
print("Average F1-score: %.4f, std: %.4f" % (average_f1, std_f1))