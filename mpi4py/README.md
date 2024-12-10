# Distributed Training with MPI4PY

The mpi4py library is being used in this code to enable distributed training and communication between processes across multiple GPUs and nodes in a high-performance computing (HPC) environment.

# Requirements

- Access to the HPC environment with necessary resources.
- Weights & Biases (wandb) token for logging.

# To run the code

1. Submit the SLURM job with the desired configuration:

```bash
sbatch slurm_X_Y_0.sh
```

Where X is the number of nodes and Y is the number of GPUs per node.

## Logging and Monitoring

You can view detailed reports of:

- Training metrics
- Evaluation scores
- Iteration time
- GPU utilization and efficiency

all accessible in your Weights & Biases dashboard.
