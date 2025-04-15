#!/bin/bash
#SBATCH --job-name=ddp_test
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=2                          # <-- now using 2 nodes
#SBATCH --ntasks-per-node=4               # one task per GPU per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=bcnx-delta-gpu
#SBATCH --time=00:20:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

# Load required modules
module purge
module load openmpi/4.1.6

# Activate your environment
eval "$(conda shell.bash hook)"
conda activate pytorch-env

# NCCL configs (recommended for cluster safety)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

# Set environment variables for multi-node DDP
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export RANK=$((SLURM_NODEID * SLURM_NTASKS_PER_NODE))

# Move to your script location
cd /projects/bcnx/kazumak2/MIMONet/ddp_training

# Launch with torchrun
torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ddp_train.py
