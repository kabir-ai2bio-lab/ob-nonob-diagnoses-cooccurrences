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

The EHRSHOT Data used for this research needs access and approval from the Stanford University ShahLab. Requirements include finishing necessary certifications and signing disclosures. 

- [Apply for Access](https://stanford.redivis.com/datasets/53gc-8rhx41kgt)

## Data Preprocessing

Once the data is available, the data preprocessing can be done by two scripts. ```project2_gcn.py``` is essential and is called by the ```preprocess_data.py``` script.

| Step  | Scripts |
| :--- | :--- |
| Run the preprocessing dcript which renders the .pkl file that will be used by GNNs  | ```preprocess_data.py```|
