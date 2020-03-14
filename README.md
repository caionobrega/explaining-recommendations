# Explaining Recommendations Through Local Surrogate Models

This repository contains the code to run the experiments present in [this paper](https://dl.acm.org/doi/10.1145/3297280.3297443)

## Setup Environment
1. Install [Anaconda Distribution for Python 3.7](https://www.anaconda.com/distribution/);
2. Create a virtual environment: `conda env create -f env.yml`;
3. Activate the virtual environment: `conda activate sac2019_env`;

NOTE:
- deactivate the virtual environment run: `conda deactivate sac2019_env`.


## Dataset

The folder dataset should have the following files (with respective headers):
- training: <user_id>\t<item_id>\t<rating>
- test: <user_id>\t<item_id>\t<rating>
- item_features: <item_id>\t<feature_name>\t<value>

The dataset folder path should be set in `experiment/utils.py`


## Folder Structure

- src folder: <br/> 
    - classes and utility functions to be able to run the experiments. 

- experiment folder: <br/> 
    - the setup subfolder has the code to generate the explanations;
    - the other scripts computes metrics