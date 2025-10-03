#!/bin/bash
#SBATCH --mem=120g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=ghx4
#SBATCH --time=03:00:00
#SBATCH --job-name=Test
#SBATCH --account=begc-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/dev_train_%j.log
#SBATCH --error=logs/dev_train_%j.log

# Initialize conda for bash
source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh

# Activate env
conda activate /u/kazumak2/.conda/envs/pytorch

echo "job is starting on $(hostname)"

#Set Python to run in unbuffered mode (-u flag)
python -u train_dropout.py
