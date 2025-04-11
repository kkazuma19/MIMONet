#!/bin/bash
#SBATCH --job-name=train_subchannel
#SBATCH --partition=gpuA40x4              
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --account=bcnx-delta-gpu
#SBATCH --time=04:00:00


# Load modules
module purge
module load openmpi/4.1.6

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch-env

# Set environment variables
export PYTHONPATH=/projects/bcnx/kazumak2/MIMONet:$PYTHONPATH

# Change to project directory
cd /projects/bcnx/kazumak2/MIMONet/Subchannel

# run training script and save log into logs directory there
python train.py > logs/train.log 2>&1

# end of script
