#!/bin/bash
#SBATCH --job-name=ddp_subchannel
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4                # one task per GPU
#SBATCH --gpus-per-node=4                  # match number of tasks
#SBATCH --cpus-per-task=8                  # threads per task (can tweak this)
#SBATCH --mem=64G
#SBATCH --account=bcnx-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

# Load required modules
module purge
module load openmpi/4.1.6

# Activate your environment
eval "$(conda shell.bash hook)"
conda activate pytorch-env

# Set environment variables for DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

# Move to your script location
cd $SLURM_SUBMIT_DIR/ddp_training

# Run your training using torchrun
torchrun --nproc_per_node=4 ddp_train.py
