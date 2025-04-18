#!/bin/bash
#SBATCH --job-name=ddp_training
#SBATCH --account=bcnx-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/ddp_training_%j.log
#SBATCH --error=logs/ddp_training_%j.err

# === MASTER NODE DISCOVERY ===
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

# === ENVIRONMENT SETUP ===
module purge
module load openmpi/4.1.6

eval "$(conda shell.bash hook)"
conda activate pytorch-env

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export LOGLEVEL=INFO

# === TRAINING COMMAND ===
cd /projects/bcnx/kazumak2/MIMONet/ddp_training

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$head_node_ip:29500" \
    ddp_train_multi_node.py

# Optional cleanup
rm -f snapshot.pt
