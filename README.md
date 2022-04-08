# RLMSAD
Reinforcement Learning-Based Model Selection for Anomaly Detection (RLMSAD)

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

