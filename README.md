# Predicting Obstetric and Non-obstetric Diagnoses Co-occurrences during Pregnancy
Contributers - Akash Singh, Samuel Infante, Dr. Seungbae Kim, Dr. Anowarul Kabir

This repository contains the training models and various accessory scripts used for the underlying research. EHR data, the graph file and any connected files to EHR are not shared. 

## Installation
```bash
# Installation of virtual environment
git clone https://github.com/kabir-ai2bio-lab/ob-nonob-diagnoses-cooccurrences.git
cd ob-nonob-diagnoses-cooccurrences
conda create -c conda-forge -p .venvs/ob_nonob_condaenv python=3.9.1 -y
conda activate .venvs/ob_nonob_condaenv
pip install -r requirements.txt

# To deactivate and remove the env
conda deactivate
conda remove --name ob_nonob_condaenv --all -y
conda remove -p .venvs/ob_nonob_condaenv --all -y

```

## Data Download

The EHRSHOT Data used for this research needs access and approval from the Stanford University ShahLab. Requirements include providing necessary training certifications and signing disclosures. 

- [Apply for Access](https://stanford.redivis.com/datasets/53gc-8rhx41kgt)

## Data Preprocessing

Once the data is available, the data preprocessing can be done by two scripts. ```project2_gcn.py```(depricated) is essential and is called by the ```preprocess_data.py``` located in graph handling directory. 

| Step  | Scripts |
| :--- | :--- |
| Run the preprocessing script which renders the .pkl file that will be used by GNNs  | ```preprocess_data.py```|
| Split the graph nodes into training/validation/test splits and save the splits. It creates a kfold split and single split.    | ```split_manager.py```  |
| Assert that the splits were generated correctly | ```split_asserts.py```|


## Hyperparameter tuning

Hyperparameters will be needed to be set accordingly. It is suggested to tune them according to respective graph structure and model architecture. We have provided the tuning steps used for selecting best hyperparameters for our models. The tuning follows a nested cross validation structure. 

- Outer loop: 5-fold (for final performance estimation)
- Inner loop: 3-fold (for hyperparameter selection)
- No outer test leakage during tuning
- Hyperparamters selected by mean inner validation AUROC

| Step  | Scripts |
| :--- | :--- |
| Create the grid jsons for each model. Sample grids provided  | See Hyperparameter tuning/param_grids folder for samples |
| Run hyperparameter tuning script. Participating models can be selected in the script    | ```tune_hyperparams_kfold.py``` |


## Training

Following 6 models are are provided that were ran consecutively on a gpu cluser. We have provided ```slurm_3_train_all.sh``` which was used to run the training and generate results. It can be modified or improvised according to respective settings. 

| Model | Script |
| GCN   | ```gcn.py``` |
| GraphSAGE  | ```graphsage.py``` |
| GAT   | ```gat.py``` |
| GCN + GraphSAGE | ```hybrid_gcn_graphsage``` |
| GCN + GAT  | ```hybrid_gcn_gat.py``` |
| GAT + GraphSAGE   | ```graphsage_gat_hybrid.py``` |

Training steps

| Step  | Scripts |
| :--- | :--- |
| Run the training script for each model, select the appropriate model | ```train_kfold_best.py```|
| Alternatively, run all of them together  | ```slurm_3_train_all.sh```  |

