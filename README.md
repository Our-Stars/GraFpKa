![Workflow](https://github.com/Our-Stars/GraFpKa/blob/master/images/Workflow.png)
# GraFpKa
A graph neural network-based interpretable platform for small molecule pKa prediction.  

## Environment Installation
* The project provides an environment file env_GraFpKa.yml. You can install the environment by running the following command:  
`conda env create -f env_GraFpKa.yml`   
  
* After that, activate the environment with:  
`conda activate env_GraFpKa`  

## Overview
* `input\`: Input data, including the training dataset, external validation set, and ionizable site file.  
* `model\`: The GraFpKa model.  
* `temp\`: A temporary folder for storing temporary data.  
* `visualization\`: Visualization results, used to store the results of interpretability analysis.  

## Usage
* If you need to train the model:  
  - You can run the following command for pretraining:   
`python pretraining.py`  
  - Then, run the following command for fine-tuning:  
`python fine_tuning.py`  
  
* If you only want to use the model:  
  - You can run the following command to predict pKa values:  
`python predicting.py`  
  - You can also run the following command for interpretability analysis:  
`python interpretability.py`
