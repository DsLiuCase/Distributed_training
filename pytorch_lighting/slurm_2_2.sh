#!/bin/bash
#SBATCH --job-name=distributed_training
#SBATCH --account=llm_workshop2024
#SBATCH -p gpu
#SBATCH --gres=gpu:2                       # Each node gets 2 GPUs
#SBATCH --nodes=2                          # Total nodes
#SBATCH --ntasks-per-node=1                # One task per node
#SBATCH --cpus-per-task=16                 # Number of CPUs per task
#SBATCH --mem=200G                         # Memory per node
#SBATCH --output=/home/dxl952/Cource/High_performance_AI/project/Llama3_1b/Distributed_training/pytorch_lighting/logs/output_2_2.log
#SBATCH --error=/home/dxl952/Cource/High_performance_AI/project/Llama3_1b/Distributed_training/pytorch_lighting/logs/error_2_2.log

# Load necessary modules
module load NVHPC/23.1-CUDA-12.0.0

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)  # Generate a random available port
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export OMP_NUM_THREADS=8

# Debugging information
echo "Python Path: $(which python)"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Node Rank: $SLURM_NODEID"

# Start distributed training
torchrun --nproc_per_node=2 \
         --nnodes=2 \
         --node_rank=$SLURM_NODEID \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         python /home/dxl952/Cource/High_performance_AI/project/Llama3_1b/Distributed_training/pytorch_lighting/train.py --num_nodes 2 --gpus_per_node 2