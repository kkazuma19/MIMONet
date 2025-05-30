#!/bin/bash
#SBATCH --job-name=subchannel_train
#SBATCH --partition=gpuA100x4              
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --account=bcnx-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

# Load modules
module purge
module load openmpi/4.1.6

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch-env

# Change to project directory
cd /projects/bcnx/kazumak2/MIMONet/LDC
#cd /projects/bcnx/kazumak2/MIMONet/Subchannel

#Set Python to run in unbuffered mode (-u flag)
python -u train.py
