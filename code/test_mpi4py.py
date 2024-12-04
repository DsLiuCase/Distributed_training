import os
import time
import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import wandb
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize wandb (only on rank 0)
if MPI.COMM_WORLD.Get_rank() == 0:
    wandb.init(project="llama-distributed", name="data-splitting-distributed")

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure MASTER_ADDR and MASTER_PORT are set
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "12345")

# Map rank to available GPUs
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
device_id = int(gpu_ids[rank % len(gpu_ids)])
torch.cuda.set_device(device_id)
device = torch.device(f"cuda:{device_id}")

# Initialize distributed environment
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=size,
    rank=rank,
)

# Load model
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch

# Load a subset of the SQuAD 2.0 dataset
NUM_TRAIN_EXAMPLES = 10  # Define the number of training examples to use
NUM_VAL_EXAMPLES = 4  # Define the number of validation examples to use

dataset = load_dataset("squad_v2")
train_data = dataset["train"].select(range(min(len(dataset["train"]), NUM_TRAIN_EXAMPLES)))
validation_data = dataset["validation"].select(range(min(len(dataset["validation"]), NUM_VAL_EXAMPLES)))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
if tokenizer.pad_token is None:  # If no padding token is set
    tokenizer.pad_token = tokenizer.eos_token  # Set eos_token as the padding token

# Preprocessing function
def preprocess_function(examples):
    questions = examples["question"]
    contexts = examples["context"]
    inputs = tokenizer(questions, contexts, truncation=True, padding=True, max_length=512)
    labels = inputs.input_ids
    return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": labels}

# Preprocess datasets
train_data = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
validation_data = validation_data.map(preprocess_function, batched=True, remove_columns=validation_data.column_names)

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data["input_ids"][idx]),
            torch.tensor(self.data["attention_mask"][idx]),
            torch.tensor(self.data["labels"][idx]),
        )

# Convert processed data into custom datasets
train_dataset = CustomDataset(train_data)
validation_dataset = CustomDataset(validation_data)

# Create DataLoader with DistributedSampler
train_sampler = DistributedSampler(train_dataset, num_replicas=size, rank=rank)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)

val_sampler = DistributedSampler(validation_dataset, num_replicas=size, rank=rank)
val_dataloader = DataLoader(validation_dataset, sampler=val_sampler, batch_size=1)

# Training loop
epochs = 300
for epoch in range(epochs):
    model.train()
    train_sampler.set_epoch(epoch)  # Shuffle data for each epoch
    epoch_loss = 0.0

    for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Allreduce gradients
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size

        optimizer.step()
        epoch_loss += loss.item()

    # Validation step (only rank 0 logs validation results)
    if rank == 0:
        model.eval()
        val_loss = 0.0
        inference_times = []  # Track inference times

        with torch.no_grad():
            for val_input_ids, val_attention_mask, val_labels in val_dataloader:
                val_input_ids = val_input_ids.to(device)
                val_attention_mask = val_attention_mask.to(device)
                val_labels = val_labels.to(device)

                # Measure inference time
                start_time = time.time()
                val_outputs = model(
                    input_ids=val_input_ids,
                    attention_mask=val_attention_mask,
                    labels=val_labels,
                )
                elapsed_time = time.time() - start_time
                inference_times.append(elapsed_time)

                val_loss += val_outputs.loss.item()

        avg_inference_time = sum(inference_times) / len(inference_times)
        wandb.log({"Epoch": epoch + 1, "Training Loss": epoch_loss / len(train_dataloader),
                   "Validation Loss": val_loss / len(val_dataloader), "Avg Inference Time (s)": avg_inference_time})

        logging.info(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_dataloader)}, "
                     f"Validation Loss: {val_loss / len(val_dataloader)}, "
                     f"Avg Inference Time: {avg_inference_time:.4f}s")

# Cleanup
dist.destroy_process_group()
if rank == 0:
    wandb.finish()