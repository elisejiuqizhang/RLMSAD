# Set current working directory to the main branch of RLMSAD
import os
import sys

from pyparsing import replaceWith
sys.path.append('/usr/local/data/elisejzh/Projects/RLMSAD') # This is the path setting on my computer, modify this according to your need
from base_detectors.USAD.usad_model import *

import random
import gym
from gym.utils import seeding
from gym import spaces

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class EnvOffline(gym.Env):
    '''Gym environment for model selection in offline setting.

    model_path: path to the pretrained models;
    list_pred_sc: the flattened list of raw predicted scores (each one being 1D numpy array) 
                    of the testing data from each model;
    list_thresholds: the list of raw anomaly thresholds from each model;'''
    
    def __init__(self, list_pred_sc, list_thresholds,list_gtruth, model_path="./base_detectors"):

        # Length of the testing data, number of models
        self.len_data = len(list_pred_sc[0])
        self.num_models = len(list_pred_sc)

        #List of ground truth labels
        self.gtruth = list_gtruth
        
        # Get the list of pretrained models
        self.model_path = model_path 
        self.model_list = [f for f in os.listdir(self.model_path) if f.endswith('.sav')]

        # Extract the model names
        self.model_names = [f.split('_')[0] for f in self.model_list]
        
        # Raw scores and thresholds of the testing data
        self.list_pred_sc = list_pred_sc
        self.list_thresholds = list_thresholds

        # Scale the raw scores/thresholds and save each scaler
        self.scaler = []
        self.list_scaled_sc = []
        self.list_scaled_thresholds = []
        for i in range(self.num_models):
            scaler_tmp = MinMaxScaler()
            self.list_scaled_sc.append(scaler_tmp.fit_transform(self.list_pred_sc[i].reshape(-1,1)))
            self.scaler.append(scaler_tmp)
            self.list_scaled_thresholds.append(scaler_tmp.transform(self.list_thresholds[i].reshape(-1,1)))

        # Extract predictions
        self.list_pred = []
        for i in range(self.num_models):
            pred_tmp = np.zeros(self.len_data)
            for length in range(self.len_data):
                if self.list_scaled_sc[i][length] > self.list_scaled_thresholds[i]:
                    pred_tmp[length] = 1
            self.list_pred.append(pred_tmp)

        # Extract prediction-concensus confidence
        self.list_concensus_conf = [] # how many models have predicted 1 (anomaly)
        for length in range(self.len_data):
            num_a_tmp=0 # number of models have predicted 1 (anomaly)
            for i in range(self.num_models):
                if self.list_scaled_sc[i][length] > self.list_scaled_thresholds[i]:
                    num_a_tmp += 1
            self.list_concensus_conf.append(num_a_tmp/self.num_models)

        # Extract distance-to-threshold confidence
        self.dist_conf=[]
        for length in range(self.len_data):
            dist_tmp = []
            for i in range(self.num_models):
                dist_tmp.append(self.list_scaled_sc[i][length] - self.list_scaled_thresholds[i])
            self.dist_conf.append(dist_tmp)
        

        # Gym settings
        self.action_space = spaces.Discrete(self.num_models) 
        # state_dim is 5 , each corresponds to scaled_sc, scaled_thresholds, pred, concensus_conf, dist_conf 
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, ), dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

class TrainEnvOffline(EnvOffline):
    '''The training environment in offline setting.

        list_gtruth: the list of ground truth labels (each one being 1D numpy array) of the 
                    testing data by each models.'''

    def __init__(self, list_pred_sc, list_thresholds, list_gtruth):
        super().__init__(list_pred_sc, list_thresholds, list_gtruth)
    
    def reset(self):
        self.pointer = 0 # Reset the pointer to the beginning of the testing data
        self.done = False
        return self._get_state()

    def step(self, action):
        '''Return:
            observation: the current state of the environment;
            reward: the reward of the action;
            done: whether the episode is over;'''

        # Get the current state
        observation = self._get_state(action)

        # Get the reward
        reward=self._get_reward(observation)

        self.pointer += 1

        # Check whether the episode is over
        if self.pointer >= self.len_data:
            self.done = True
        else:
            self.done = False

        return observation, reward, self.done, {}

    def _get_state(self,action=None):

        '''Return:
            observation: the current state of the environment.'''

        if self.pointer==0: # If the pointer is at the beginning of the testing data
            action=random.randint(0,self.num_models-1) # Randomly select a model

        # Get the current state
        observation = np.zeros(5) # 5 dims - scaled scores, scaled thresholds, labels, concensus_conf, dist_conf
        observation[0] = self.list_scaled_sc[action][self.pointer]
        observation[1] = self.list_scaled_thresholds[action]
        observation[2] = self.list_pred[action][self.pointer]
        if observation[2]==1:
            observation[3] = self.list_concensus_conf[self.pointer]
        else:
            observation[3] = 1-self.list_concensus_conf[self.pointer]
        observation[4] = self.dist_conf[self.pointer][action]

        return observation

    def _get_reward(self,observation):
        '''Return:
            reward: the reward of the action.'''

        # Get the reward
        if self.gtruth[self.pointer]==1: # If the ground truth is 1 anomaly
            if observation[2]==1: # If the model predicts 1 anomaly correctly - True Positive (TP)
                reward = 1
            else: # If the model predicts 0 normal incorrectly - False Negative (FN)
                reward = -1.5
        else: # If the ground truth is 0 normal
            if observation[2]==1: # If the model predicts 1 anomaly incorrectly - False Positive (FP)
                reward = -0.5
            else: # If the model predicts 0 normal correctly - True Negative (TN)
                reward = 0.1

        return reward


def eval_model(model,env):
    '''Evaluate the model on the environment.

        model: the model to be evaluated;
        env: the environment to be evaluated on.
        
        Return:
            precision: the precision of the model;
            recall: the recall of the model;
            f1: the f1 score of the model;
            conf_matrix: the confusion matrix of the model, comparing it with the ground truth;
            preds: the list of predictions of the model.'''

    # The ground truth labels
    gtruth = env.gtruth

    # Reset the environment
    observation = env.reset()

    # Evaluate the model - get predicted labels and total reward
    preds = []
    while True:
        action = model.predict(observation)
        observation, reward, done, _ = env.step(action[0]) # action[0] is the index of the action, action is a tuple
        preds.append(observation[2])
        if done:
            break
    
    prec=precision_score(gtruth,preds,pos_label=1)
    rec=recall_score(gtruth,preds,pos_label=1)
    f1=f1_score(gtruth,preds,pos_label=1)
    conf_matrix=confusion_matrix(gtruth,preds,labels=[0,1])

    return prec,rec,f1,conf_matrix, preds



'''Below are the two environments for the ablation study - probing the effect of two confidence scores'''

class EnvOffline_consensus_conf(gym.Env):
    '''Gym environment for model selection in offline setting, only use the consensus confidence,
    disgard the distance to threshold.

    model_path: path to the pretrained models;
    list_pred_sc: the flattened list of raw predicted scores (each one being 1D numpy array) 
                    of the testing data from each model;
    list_thresholds: the list of raw anomaly thresholds from each model;'''
    
    def __init__(self, list_pred_sc, list_thresholds,list_gtruth, model_path="./base_detectors"):

        # Length of the testing data, number of models
        self.len_data = len(list_pred_sc[0])
        self.num_models = len(list_pred_sc)

        #List of ground truth labels
        self.gtruth = list_gtruth
        
        # Get the list of pretrained models
        self.model_path = model_path 
        self.model_list = [f for f in os.listdir(self.model_path) if f.endswith('.sav')]

        # Extract the model names
        self.model_names = [f.split('_')[0] for f in self.model_list]
        
        # Raw scores and thresholds of the testing data
        self.list_pred_sc = list_pred_sc
        self.list_thresholds = list_thresholds

        # Scale the raw scores/thresholds and save each scaler
        self.scaler = []
        self.list_scaled_sc = []
        self.list_scaled_thresholds = []
        for i in range(self.num_models):
            scaler_tmp = MinMaxScaler()
            self.list_scaled_sc.append(scaler_tmp.fit_transform(self.list_pred_sc[i].reshape(-1,1)))
            self.scaler.append(scaler_tmp)
            self.list_scaled_thresholds.append(scaler_tmp.transform(self.list_thresholds[i].reshape(-1,1)))

        # Extract predictions
        self.list_pred = []
        for i in range(self.num_models):
            pred_tmp = np.zeros(self.len_data)
            for length in range(self.len_data):
                if self.list_scaled_sc[i][length] > self.list_scaled_thresholds[i]:
                    pred_tmp[length] = 1
            self.list_pred.append(pred_tmp)

        # Extract prediction-concensus confidence
        self.list_concensus_conf = [] # how many models have predicted 1 (anomaly)
        for length in range(self.len_data):
            num_a_tmp=0 # number of models have predicted 1 (anomaly)
            for i in range(self.num_models):
                if self.list_scaled_sc[i][length] > self.list_scaled_thresholds[i]:
                    num_a_tmp += 1
            self.list_concensus_conf.append(num_a_tmp/self.num_models)

        # Gym settings
        self.action_space = spaces.Discrete(self.num_models) 
        # state_dim is 4 (reduced) , each corresponds to scaled_sc, scaled_thresholds, pred, concensus_conf
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float32) 
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

class TrainEnvOffline_consensus_conf(EnvOffline_consensus_conf):
    '''The training environment in offline setting with only prediction consensus confidence.

        list_gtruth: the list of ground truth labels (each one being 1D numpy array) of the 
                    testing data by each models.'''

    def __init__(self, list_pred_sc, list_thresholds, list_gtruth):
        super().__init__(list_pred_sc, list_thresholds, list_gtruth)
    
    def reset(self):
        self.pointer = 0 # Reset the pointer to the beginning of the testing data
        self.done = False
        return self._get_state()

    def step(self, action):
        '''Return:
            observation: the current state of the environment;
            reward: the reward of the action;
            done: whether the episode is over;'''

        # Get the current state
        observation = self._get_state(action)

        # Get the reward
        reward=self._get_reward(observation)

        self.pointer += 1

        # Check whether the episode is over
        if self.pointer >= self.len_data:
            self.done = True
        else:
            self.done = False

        return observation, reward, self.done, {}

    def _get_state(self,action=None):

        '''Return:
            observation: the current state of the environment.'''

        if self.pointer==0: # If the pointer is at the beginning of the testing data
            action=random.randint(0,self.num_models-1) # Randomly select a model

        # Get the current state
        observation = np.zeros(4) # 5 dims - scaled scores, scaled thresholds, labels, concensus_conf, dist_conf
        observation[0] = self.list_scaled_sc[action][self.pointer]
        observation[1] = self.list_scaled_thresholds[action]
        observation[2] = self.list_pred[action][self.pointer]
        if observation[2]==1:
            observation[3] = self.list_concensus_conf[self.pointer]
        else:
            observation[3] = 1-self.list_concensus_conf[self.pointer]

        return observation

    def _get_reward(self,observation):
        '''Return:
            reward: the reward of the action.'''

        # Get the reward
        if self.gtruth[self.pointer]==1: # If the ground truth is 1 anomaly
            if observation[2]==1: # If the model predicts 1 anomaly correctly - True Positive (TP)
                reward = 1
            else: # If the model predicts 0 normal incorrectly - False Negative (FN)
                reward = -1.5
        else: # If the ground truth is 0 normal
            if observation[2]==1: # If the model predicts 1 anomaly incorrectly - False Positive (FP)
                reward = -0.4
            else: # If the model predicts 0 normal correctly - True Negative (TN)
                reward = 0.1

        return reward




class EnvOffline_dist_conf(gym.Env):
    '''Gym environment for model selection in offline setting.

    model_path: path to the pretrained models;
    list_pred_sc: the flattened list of raw predicted scores (each one being 1D numpy array) 
                    of the testing data from each model;
    list_thresholds: the list of raw anomaly thresholds from each model;'''
    
    def __init__(self, list_pred_sc, list_thresholds,list_gtruth, model_path="./base_detectors"):

        # Length of the testing data, number of models
        self.len_data = len(list_pred_sc[0])
        self.num_models = len(list_pred_sc)

        #List of ground truth labels
        self.gtruth = list_gtruth
        
        # Get the list of pretrained models
        self.model_path = model_path 
        self.model_list = [f for f in os.listdir(self.model_path) if f.endswith('.sav')]

        # Extract the model names
        self.model_names = [f.split('_')[0] for f in self.model_list]
        
        # Raw scores and thresholds of the testing data
        self.list_pred_sc = list_pred_sc
        self.list_thresholds = list_thresholds

        # Scale the raw scores/thresholds and save each scaler
        self.scaler = []
        self.list_scaled_sc = []
        self.list_scaled_thresholds = []
        for i in range(self.num_models):
            scaler_tmp = MinMaxScaler()
            self.list_scaled_sc.append(scaler_tmp.fit_transform(self.list_pred_sc[i].reshape(-1,1)))
            self.scaler.append(scaler_tmp)
            self.list_scaled_thresholds.append(scaler_tmp.transform(self.list_thresholds[i].reshape(-1,1)))

        # Extract predictions
        self.list_pred = []
        for i in range(self.num_models):
            pred_tmp = np.zeros(self.len_data)
            for length in range(self.len_data):
                if self.list_scaled_sc[i][length] > self.list_scaled_thresholds[i]:
                    pred_tmp[length] = 1
            self.list_pred.append(pred_tmp)

        # Extract distance-to-threshold confidence
        self.dist_conf=[]
        for length in range(self.len_data):
            dist_tmp = []
            for i in range(self.num_models):
                dist_tmp.append(self.list_scaled_sc[i][length] - self.list_scaled_thresholds[i])
            self.dist_conf.append(dist_tmp)
        

        # Gym settings
        self.action_space = spaces.Discrete(self.num_models) 
        # state_dim is 4 , each corresponds to scaled_sc, scaled_thresholds, pred, dist_conf 
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

class TrainEnvOffline_dist_conf(EnvOffline_dist_conf):
    '''The training environment in offline setting.

        list_gtruth: the list of ground truth labels (each one being 1D numpy array) of the 
                    testing data by each models.'''

    def __init__(self, list_pred_sc, list_thresholds, list_gtruth):
        super().__init__(list_pred_sc, list_thresholds, list_gtruth)
    
    def reset(self):
        self.pointer = 0 # Reset the pointer to the beginning of the testing data
        self.done = False
        return self._get_state()

    def step(self, action):
        '''Return:
            observation: the current state of the environment;
            reward: the reward of the action;
            done: whether the episode is over;'''

        # Get the current state
        observation = self._get_state(action)

        # Get the reward
        reward=self._get_reward(observation)

        self.pointer += 1

        # Check whether the episode is over
        if self.pointer >= self.len_data:
            self.done = True
        else:
            self.done = False

        return observation, reward, self.done, {}

    def _get_state(self,action=None):

        '''Return:
            observation: the current state of the environment.'''

        if self.pointer==0: # If the pointer is at the beginning of the testing data
            action=random.randint(0,self.num_models-1) # Randomly select a model

        # Get the current state
        observation = np.zeros(4) # 4 dims - scaled scores, scaled thresholds, labels, dist_conf
        observation[0] = self.list_scaled_sc[action][self.pointer]
        observation[1] = self.list_scaled_thresholds[action]
        observation[2] = self.list_pred[action][self.pointer]
        observation[3] = self.dist_conf[self.pointer][action]

        return observation

    def _get_reward(self,observation):
        '''Return:
            reward: the reward of the action.'''

        # Get the reward
        if self.gtruth[self.pointer]==1: # If the ground truth is 1 anomaly
            if observation[2]==1: # If the model predicts 1 anomaly correctly - True Positive (TP)
                reward = 1
            else: # If the model predicts 0 normal incorrectly - False Negative (FN)
                reward = -1.5
        else: # If the ground truth is 0 normal
            if observation[2]==1: # If the model predicts 1 anomaly incorrectly - False Positive (FP)
                reward = -0.4
            else: # If the model predicts 0 normal correctly - True Negative (TN)
                reward = 0.1

        return reward