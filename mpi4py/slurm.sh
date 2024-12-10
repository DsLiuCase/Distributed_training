#!/bin/bash
#SBATCH --job-name=distributed_training       # Job name
#SBATCH --nodes=2                             # Number of nodes
#SBATCH --ntasks-per-node=2                   # Tasks per node
#SBATCH --gres=gpu:2                          # GPUs per node
#SBATCH --cpus-per-task=10                    # CPUs per task
#SBATCH --time=1-40:00                        # Maximum runtime
#SBATCH --account=llm_workshop2024            # Account name
#SBATCH -p gpu                                # GPU partition
#SBATCH --output=output.log                # Standard output log
#SBATCH --error=error.log                  # Standard error log
#SBATCH --mem=200G                             # Memory per node

module load OpenMPI
source activate /home/dxl952/.conda/envs/hpc  # Activate Conda environment
echo "Using Python: $(which python)"

# Set distributed training variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
echo "slurm_procid: $SLURM_PROCID"
# Debugging information
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT, WORLD_SIZE: $WORLD_SIZE, RANK: $RANK"

# Run the distributed training script
mpirun --mca pml ucx --mca btl ^vader,tcp,self -n $WORLD_SIZE \
       python /home/dxl952/Cource/High_performance_AI/project/Llama3_1b/Distributed_training/code/main.py
