# Distributed Training with PyTorch Lightning

This project demonstrates the implementation of distributed training using PyTorch Lightning as a baseline framework. We trained the meta-llama/Llama-3.2-1B-Instruct on a subset of the SQuAD2.0 dataset within a high-performance computing (HPC) environment.

## Requirements

- Access to the HPC environment with necessary resources.
- Hugging Face token with access to `meta-llama/Llama-3.2-1B-Instruct`.
- Weights & Biases (wandb) token for logging.

## To run the code

1. Prepare the environment

```bash
# load python
module load Python/3.10
module load NVHPC/23.1-CUDA-12.0.0

# create virtual environment
python -m venv <path_to_your_virtual_environment>
source <path_to_your_virtual_environment>/bin/activate

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt
```

2. Prepare the Dataset

```bash
python create_squad_dataset.py
```

2. Update the SLURM File

- Replace `<your-wandb-api-key>` with your actual Weights & Biases API key.
- Replace `<your-hf-api-key>` with your actual Hugging Face API key.
- Update `<path_to_your_virtual_environment>` with the full path to your virtual environment.

3. Submit the SLURM job with the desired configuration:

```bash
sbatch slurm_X_Y.sh
```

## Logging and Monitoring

You can view detailed reports of:

- Training metrics
- Evaluation scores
- Iteration time
- GPU utilization and efficiency

all accessible in your Weights & Biases dashboard.

## Acknowledgments

- [huggingface-projects/llama-2-7b-chat](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat)
- [PyTorch Lightning](https://lightning.ai/lightning-ai/studios/finetune-an-llm-with-pytorch-lightning?view=public&section=featured&tab=overview)
- [llama-squad](https://github.com/huggingface/llama-squad)
