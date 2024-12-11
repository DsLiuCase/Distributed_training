#!/bin/bash
#SBATCH --job-name=distributed_training       # Job name
#SBATCH --gres=gpu:1                          # GPUs per node
#SBATCH --cpus-per-task=10                    # CPUs per task
#SBATCH --time=1-40:00                        # Maximum runtime
#SBATCH --account=llm_workshop2024            # Account name
#SBATCH -p gpu                                # GPU partition
#SBATCH --output=/home/dxl952/Cource/High_performance_AI/project/Llama3_1b/Distributed_training/pytorch_lighting/logs/output_1_1.log                # Standard output log
#SBATCH --error=/home/dxl952/Cource/High_performance_AI/project/Llama3_1b/Distributed_training/pytorch_lighting/logs/error_1_1.log                  # Standard error log
#SBATCH --mem=200G             


module load OpenMPI
source activate /home/dxl952/.conda/envs/hpc  # Activate Conda environment
echo "Using Python: $(which python)"




# # Export environment variables
# export WANDB_API_KEY="<your-wandb-api-key>"   # Replace with your actual WandB API key
# export HF_TOKEN="<your-hf-api-key>"          # Replace with your actual Hugging Face API key

# Set PyTorch Distributed configuration
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS="ignore"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Run the script
python train.py --num_nodes 1 --gpus_per_node 1 