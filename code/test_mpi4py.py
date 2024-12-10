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
from tqdm import tqdm
import argparse



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


branch = 2  # Set branch (1 for all ranks, 2 for only rank 0)
node = 2  # Set number of nodes
gpu = 1  # Set number of GPUs in each node
# Initialize wandb (only on rank 0)
text = 'inference in rank0' if branch == 2 else 'inference in all ranks' 


if MPI.COMM_WORLD.Get_rank() == 0:
    wandb.init(project="llama-distributed", name=f"{node} nodes, {gpu} gpu in each node; {text}")

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
PID  = os.getpid()
print(f"rank = {rank}, size = {size}, PID = {PID}")

# Map rank to available GPUs
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
print("gpu_ids = ", gpu_ids)
device_id = int(gpu_ids[rank % len(gpu_ids)])
print("device_id = ", device_id)
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

# Load a subset of the SQuAD 2.0 dataset
NUM_TRAIN_EXAMPLES = 500  # Define the number of training examples to use
NUM_VAL_EXAMPLES = 50  # Define the number of validation examples to use

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


# Preprocess datasets
train_data = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
validation_data = validation_data.map(preprocess_function, batched=True, remove_columns=validation_data.column_names)

# Convert processed data into custom datasets
train_dataset = CustomDataset(train_data)
validation_dataset = CustomDataset(validation_data)

# Create DataLoader with DistributedSampler
train_sampler = DistributedSampler(train_dataset, num_replicas=size, rank=rank)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)

val_sampler = DistributedSampler(validation_dataset, num_replicas=size, rank=rank)
val_dataloader = DataLoader(validation_dataset, sampler=val_sampler, batch_size=1)


# Validation step
def validate_branch_1():
    """Branch 1: All ranks perform validation and results are aggregated."""
    model.eval()
    val_loss = 0.0

    # Start timing for the entire validation dataset
    start_time = time.time()
    print("the length of val_dataloader is ", len(val_dataloader))
    with torch.no_grad():
        for val_input_ids, val_attention_mask, val_labels in val_dataloader:
            val_input_ids = val_input_ids.to(device)
            val_attention_mask = val_attention_mask.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(
                input_ids=val_input_ids,
                attention_mask=val_attention_mask,
                labels=val_labels,
            )
            val_loss += val_outputs.loss.item()

    # End timing for the entire validation dataset
    total_inference_time = time.time() - start_time

    # Calculate and synchronize results across all ranks
    avg_val_loss = torch.tensor(val_loss / len(val_dataloader)).to(device)
    total_inference_time = torch.tensor(total_inference_time).to(device)
    dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_inference_time, op=dist.ReduceOp.SUM)

    avg_val_loss /= size
    total_inference_time /= size  # Averaged across ranks

    return avg_val_loss.item(), total_inference_time.item()

def validate_branch_2():
    """Branch 2: Only rank 0 performs validation for the entire dataset."""
    if rank == 0:
        model.eval()
        val_loss = 0.0

        # Start timing for the entire validation dataset
        start_time = time.time()
        # assert len(val_dataloader) == validation_dataset.__len__()/size, "the length of val_dataloader is " + str(len(val_dataloader)) + ", the length of validation_dataset is " + str(validation_dataset.__len__()/size)
        with torch.no_grad():
            for val_input_ids, val_attention_mask, val_labels in val_dataloader:
                val_input_ids = val_input_ids.to(device)
                val_attention_mask = val_attention_mask.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(
                    input_ids=val_input_ids,
                    attention_mask=val_attention_mask,
                    labels=val_labels,
                )
                val_loss += val_outputs.loss.item()

        # End timing for the entire validation dataset
        total_inference_time = time.time() - start_time

        avg_val_loss = val_loss / len(val_dataloader)
        return avg_val_loss, total_inference_time
    else:
        return None, None
# Main training loop
epochs = 100

print (f"rank = {rank}  # Set rank (0 for master, 1 for worker)")
print(f"size = {size}  # Set number of processes")
print(f"validation strategy is {branch}  # Set branch (1 for all ranks, 2 for only rank 0)")
for epoch in (range(epochs)):
    epoch_start_time = time.time()

    model.train()
    train_sampler.set_epoch(epoch)  # Shuffle data for each epoch
    epoch_loss = 0.0

    for batch_input_ids, batch_attention_mask, batch_labels in (train_dataloader ):
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

    # Calculate epoch time
    epoch_time = time.time() - epoch_start_time
    
    # Perform validation based on the selected branch
    if branch == 1:
        avg_val_loss, avg_inference_time = validate_branch_1()
    else:
        avg_val_loss, avg_inference_time = validate_branch_2()

    # Log results
    if rank == 0:
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": epoch_loss / len(train_dataloader),
            "Validation Loss": avg_val_loss,
            "Avg Inference Time (s)": avg_inference_time,
            "Epoch Time (s)": epoch_time,
        })

        logging.info(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_dataloader)}, "
                     f"Validation Loss: {avg_val_loss}, "
                     f"Avg Inference Time: {avg_inference_time:.4f}s, "
                     f"Epoch Time: {epoch_time:.4f}s")

# Cleanup
dist.destroy_process_group()
if rank == 0:
    wandb.finish()
    
print("Done!")

def main(branch, number_of_nodes, number_of_gpus):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    PID  = os.getpid()
    print(f"rank = {rank}, size = {size}, PID = {PID}")

    # Ensure MASTER_ADDR and MASTER_PORT are set
    os.environ.setdefault("MASTER_ADDR", "  


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Distributed training with MPI and PyTorch")
  parser.add_argument("--branch", type=int, default=1, help="Branch to execute (1 or 2)")
  parser.add_argument("--node", type=int, default=2, help="Number of nodes")
  parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs in each node")
  args = parser.parse_args()
  number_of_nodes = args.node
  number_of_gpus = args.gpu
  branch = args.branch
  print(f"branch = {branch}, node = {number_of_nodes}, gpu = {number_of_gpus}")
  main(branch, number_of_nodes, number_of_gpus)
    
    
  
