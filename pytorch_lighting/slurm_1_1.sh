#!/bin/bash
#SBATCH --job-name=distributed_training
#SBATCH -p markov_gpu
#SBATCH -A sxk1942_csds451
#SBATCH --gres=gpu:1


# Load required modules
module load NVHPC/23.1-CUDA-12.0.0
module load Python/3.10

# Backup
# python -m venv <path_to_your_virtual_environment>
# source <path_to_your_virtual_environment>/bin/activate
# Install Python dependencies
# pip install --no-cache-dir -r requirements.txt

# Activate virtual environment
source <path_to_your_virtual_environment>/bin/activate

# Export environment variables
export WANDB_API_KEY="<your-wandb-api-key>"   # Replace with your actual WandB API key
export HF_TOKEN="<your-hf-api-key>"          # Replace with your actual Hugging Face API key

# Set PyTorch Distributed configuration
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS="ignore"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Run the script
python train.py \
    --bf16 \
    --max_seq_length=4096 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --max_steps=65000 \
    --merge_and_push \
    --save_steps=1000 \
    --lr_scheduler_type=cosine \
    --learning_rate=1e-6 \
    --warmup_ratio=0.03
    --node_count=1
    --gpu_per_node=1
