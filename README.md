
# GraFpKa
A graph neural network-based interpretable platform for small molecule pKa prediction. 
  
![Workflow](https://github.com/Our-Stars/GraFpKa/blob/master/images/Workflow.png)  

## Overview
* `input\`: Input data, including the training dataset, external validation set, and ionizable site file.  
* `model\`: The GraFpKa model.
* `results\`:Used to store prediction results.
* `temp\`: A temporary folder for storing temporary data.  
* `visualization\`: Visualization results, used to store the results of interpretability analysis.
  
## Environment Installation
* The project provides an environment file env_GraFpKa.yml. You can install the environment by running the following command:  
`conda env create -f env_GraFpKa.yml`   
  
* After that, activate the environment with:  
`conda activate env_GraFpKa`  
  
## Training 
* If you need to train the model:  
  - You can run the following command for pretraining:   
`python pretraining.py`  
  - Then, run the following command for fine-tuning:  
`python fine_tuning.py`  

## pKa Prediction
### General Command Format 
&nbsp;&nbsp;&nbsp;&nbsp;`python predicting.py [--s SMILES...] [--i INPUT_FILE] [--o OUTPUT_FILE]`  
### Parameter explanation:  
&nbsp;&nbsp;&nbsp;&nbsp;`--s SMILES...`: Input multiple SMILES strings directly via the command line. (Mutually exclusive with `--i`).  
&nbsp;&nbsp;&nbsp;&nbsp;`--i INPUT_FILE`: Provide a CSV file containing SMILES strings (one SMILES per line, no header). (Mutually exclusive with `--s`).  
&nbsp;&nbsp;&nbsp;&nbsp;`--o OUTPUT_FILE`: Optional. Specify the output CSV file path for saving pKa predictions. If not provided, the default output path is `results/predictions.csv. ` 
### Examples:  
Predicting pKa values with SMILES input directly:  
&nbsp;&nbsp;&nbsp;&nbsp;`python predicting.py --s "CC(=O)OC1=CC=CC=C1C(=O)O" "Cc1ccccc1NN=C(C#N)C#N" "COc1cccc(S(N)(=O)=O)c1"`  
Predicting pKa values with a CSV input file:  
&nbsp;&nbsp;&nbsp;&nbsp;`python predicting.py --i input/input_example.csv --o results/predictions.csv`  

## Interpretability Analysis  
### General Command Format:
&nbsp;&nbsp;&nbsp;&nbsp;`python interpretability.py [--s SMILES...] [--i INPUT_FILE] [--o OUTPUT_DIR] [--t THRESHOLD]`  
### Parameter explanation:  
&nbsp;&nbsp;&nbsp;&nbsp;`--s SMILES...`: Input multiple SMILES strings directly via the command line.  (Mutually exclusive with `--i`).  
&nbsp;&nbsp;&nbsp;&nbsp;`--i INPUT_FILE`: Provide a CSV file containing SMILES strings (one SMILES per line, no header). (Mutually exclusive with `--s`).  
&nbsp;&nbsp;&nbsp;&nbsp;`--o OUTPUT_FILE`: Optional. Specify the output directory to save interpretability analysis images. The directory must end with `/`. If not provided, the default output path is `visualization/`.  
&nbsp;&nbsp;&nbsp;&nbsp;`--t THRESHOLD`: Optional.Specify the threshold for the color bar in the interpretability analysis. Default is -1 for automatic thresholding.  
### Examples:  
Performing interpretability analysis with SMILES input directly:  
&nbsp;&nbsp;&nbsp;&nbsp;`python interpretability.py --s "CC(=O)OC1=CC=CC=C1C(=O)O" "Cc1ccccc1NN=C(C#N)C#N" "COc1cccc(S(N)(=O)=O)c1" --t -1`  
Performing interpretability analysis with a CSV input file:  
&nbsp;&nbsp;&nbsp;&nbsp;`python interpretability.py --i input/input_example.csv --o visualization/`  
