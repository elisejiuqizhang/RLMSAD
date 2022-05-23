# RLMSAD: Reinforcement Learning-Based Model Selection for Anomaly Detection
This is the implementation for Reinforcement Learning-Based Model Selection for Anomaly Detection (RLMSAD). 

Paper title: Time Series Anomaly Detection via Reinforcement Learning-Based Model Selection

## Requirements
 * python 3.7.11
 * torch 1.10.0
 * cudatoolkit 11.3
 * numpy 1.21.2
 * sklearn 1.0.2
 * pandas 1.3.5
 * matplotlib 3.5.1
 * seaborn 0.11.2
 * pyod 0.9.8
 * gym 0.21.0
 * stable-baselines3 1.5.0


## Requesting Access to the Datasets and Usage

SWaT datasets is collected by “iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design”. If you intend to publish paper using it, you have to first request access through their official website https://itrust.sutd.edu.sg/itrust-labs_datasets/ and give explicit credit to their lab.

* SWaT dataset: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat

### Dataset Usage
The data files are too large to be uploaded to the repo. After your access to the datasets has been approved, you should create a folder ```raw_input/SWaT``` under the master branch, then download the corresponding CSV files into it. (What I was using when testing was the Attack_v0 and Normal_v1 under "SWaT A1&A2 Dec 2015 - Physical")

## Implementation Workflow

Check and modify the path in ```sys.path.append('/usr/local/data/elisejzh/Projects/RLMSAD')``` according to your path settings for related scripts.

### 1. Data Preprocessing
These functions are already incorporated into a pre-defined module "data_process.py". (There is no need to run it individually as it will be loaded into and run by other scripts.)
### 2. Pretrain Base Detectors
The five candidate base anomaly detectors include
| Base Models                                                                            	| Paper Source                                                                                   	| Implementations                       	|
|----------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------	|---------------------------------------	|
| ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions 	| https://arxiv.org/abs/2201.00382                                                               	| https://github.com/yzhao062/pyod      	|
| COPOD: Copula-Based Outlier Detection                                                  	| https://arxiv.org/abs/2009.09463                                                               	| https://github.com/yzhao062/pyod      	|
| One-Class Support Vector Machine (SVM)                                                 	| https://www.jmlr.org/papers/v12/pedregosa11a.html                                              	| sklearn.linear_model.SGDOneClassSVM() 	|
| Isolation Forest (iForest)                                                             	| https://ieeexplore.ieee.org/document/4781136 https://www.jmlr.org/papers/v12/pedregosa11a.html 	| sklearn.ensemble.IsolationForest()    	|
| USAD: UnSupervised Anomaly Detection on Multivariate Time Series                       	| https://dl.acm.org/doi/10.1145/3394486.3403392                                                 	| https://github.com/manigalati/usad    	|

ECOD and COPOD are from the pyod package. SGD one-class SVM (OSVM) and Isolation Forest (iForest) are from sklearn. USAD implementation are from the authors' orginal repository.

#### 2.1 To pretrain each base detector and save the model as '.sav' file using pickle
From terminal. set the current working directory to the root of master branch of RLMSAD: 

  ```$ cd RLMSAD```

Pretrain and save each model on training data:

 ```$ python base_detectors/PyOD_models/ECOD_train.py```
 
 ```$ python base_detectors/PyOD_models/COPOD_train.py```
 
 ```$ python base_detectors/sklearn_models/OSVM_train.py```
 
 ```$ python base_detectors/sklearn_models/iForest_train.py```
 
 ```$ python base_detectors/USAD/usad_train.py```

#### 2.2 Run the pretrained model on test data:
The pretrained models will be saved under the root of master branch after executing the above training scripts.
Run the following scripts for evaluating the base models.

 ```$ python base_detectors/PyOD_models/eval_ECOD.py```
 
 ```$ python base_detectors/PyOD_models/eval_COPOD.py```
 
 ```$ python base_detectors/sklearn_models/eval_OSVM.py```

 ```$ python base_detectors/sklearn_models/eval_iForest.py```
 
 ```$ python base_detectors/USAD/eval_usad.py```
 
 You can vary the model threshold by changing the contamination rate ```contamination``` in function ```raw_thredholds(raw_scores, contamination)``` to tune the base model performance. 
 



### 3. Run RL Model Selector

To change the reward setting in gym environment, go to ```RL_settings/env.py```. 

To change the RL algorithm for training the policy, go to ```RL_settings/run_RLMS_offline.py``` to load a different algorithm from stable-baselines3 package. 

Set the threshold for each base detector in script ```RL_settings/run_RLMS_offline.py``` (ideally based on your tuning results on the pretrained models).

To train the policy, execute the following command in terminal:

```$ python RL_settings/run_RLMS_offline.py```
