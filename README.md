# RLMSAD: Reinforcement Learning-Based Model Selection for Anomaly Detection
This is the implementation for Reinforcement Learning-Based Model Selection for Anomaly Detection (RLMSAD).

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
The data files are too large to be uploaded to the repo. After your access to the datasets has been approved, you should create a folder "raw_input/SWaT" under the master branch, then download the corresponding CSV files into it. (What I was using when testing was the Attack_v0 and Normal_v1 under "SWaT A1&A2 Dec 2015 - Physical")

## Implementation Workflow

### 1. Data Preprocessing
These functions are already incorporated into a pre-defined module "data_process.py". There is no need to run this it individually as it will be loaded into and run by other scripts.
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
  $ cd RLMSAD

Pretrain and save each model:
  $ python base_detectors/PyOD_models/ECOD_train.py
  $ python base_detectors/PyOD_models/COPOD_train.py
  $ python base_detectors/sklearn_models/OSVM_train.py
  $ python base_detectors/sklearn_models/iForest_train.py

#### 2.2 


### 3. Run RL Model Selector
