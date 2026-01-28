#!/bin/bash
#SBATCH --job-name=revamp_train_all
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:TitanV:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# STEP 3: Train all 5 models with best hyperparameters (GPU)
# Uses pre-generated splits and example hyperparameters

set -e

echo "========================================="
echo "REVAMP: Train All Models (K-Fold)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================="

# Check disk space
AVAILABLE=$(df -h . | tail -1 | awk '{print $4}')
echo "Available disk space: $AVAILABLE"

# Activate environment
source ~/.bashrc
source ~/envs/ehrshot_gnn_env/bin/activate

# Navigate to REVAMP directory
cd /general/akashsingh/ehrshot-network-analysis_project1_akash_samuel/REVAMP

# Create directories
mkdir -p runs/final_kfold
mkdir -p logs

# Verify environment
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

# Check GPU
nvidia-smi

# Check that splits exist
if [ ! -f "splits/kfold_k5_seed42.npz" ]; then
    echo "ERROR: Splits not found! Run slurm_1_generate_splits.sh first."
    exit 1
fi

# Train all 6 models
for model in gcn graphsage gat hybrid_gcn_gat hybrid_gcn_graphsage graphsage_gat_hybrid; do
    echo ""
    echo "========================================="
    echo "Training: $model"
    echo "========================================="
    
    python train_kfold_best.py \
        --pkl ../preprocessed_data/pregnancy_graph_data.pkl \
        --splits splits/kfold_k5_seed42.npz \
        --model $model \
        --hyperparams_file hyperparams/${model}_best.json \
        --max_epochs 300 \
        --patience 40 \
        --out_dir runs/final_kfold \
        --seed 42
    
    echo "✓ $model training complete"
done

echo ""
echo "========================================="
echo "All models trained successfully!"
echo "Output directory: runs/final_kfold"
echo "Finished: $(date)"
echo "========================================="

# List results
echo ""
echo "Results:"
ls -lh runs/final_kfold/
