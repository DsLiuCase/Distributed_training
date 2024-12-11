#!/bin/bash
#SBATCH --job-name=distributed_training       # Job name
#SBATCH --gres=gpu:4                        # GPUs per node
#SBATCH --cpus-per-task=10                    # CPUs per task
#SBATCH --time=1-40:00                        # Maximum runtime
#SBATCH --account=llm_workshop2024            # Account name
#SBATCH -p gpu                                # GPU partition
#SBATCH --output=/home/dxl952/Cource/High_performance_AI/project/Llama3_1b/Distributed_training/pytorch_lighting/logs/output_1_4.log                # Standard output log
#SBATCH --error=/home/dxl952/Cource/High_performance_AI/project/Llama3_1b/Distributed_training/pytorch_lighting/logs/error_1_4.log                  # Standard error log
#SBATCH --mem=200G                             # Memory per node



module load OpenMPI
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc
echo "Using Python: $(which python)"
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29500
# # Export environment variables
# export WANDB_API_KEY="<your-wandb-api-key>"   # Replace with your actual WandB API key
# export HF_TOKEN="<your-hf-api-key>"          # Replace with your actual Hugging Face API key

# Set PyTorch Distributed configuration
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS="ignore"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker

# Run the script
export PL_TRAINER_DEBUG=1
python train.py --num_nodes 1 --gpus_per_node 4