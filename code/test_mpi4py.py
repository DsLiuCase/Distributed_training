import os
import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.utils.data import DataLoader, TensorDataset
from transformers import LlamaForCausalLM, LlamaTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# # Ensure MASTER_ADDR and MASTER_PORT are set
# if "MASTER_ADDR" not in os.environ:
#     os.environ["MASTER_ADDR"] = "127.0.0.1"
# if "MASTER_PORT" not in os.environ:
#     os.environ["MASTER_PORT"] = "12345"

# Map rank to available GPUs
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

if not gpu_ids or rank >= len(gpu_ids):
    raise RuntimeError(f"Rank {rank} cannot find a valid GPU in CUDA_VISIBLE_DEVICES.")
torch.cuda.set_device(int(gpu_ids[rank % len(gpu_ids)]))
device = torch.device("cuda")

# Initialize distributed environment
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=size,
    rank=rank,
)

# Load Llama model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
model = LlamaForCausalLM.from_pretrained("hf-internal-testing/llama-model").to(device)

# Wrap model for distributed training
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(gpu_ids[rank % len(gpu_ids)])])

# Prepare dataset
inputs = ["Hello, world!", "Fine-tuning Llama is fun!", "Distributed training is powerful."]
targets = ["Hello, world!", "Fine-tuning Llama is fun!", "Distributed training is powerful."]

# Tokenize data
inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
targets = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")

dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, targets.input_ids)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Validation set
val_inputs = tokenizer(["Validation example 1", "Validation example 2"], padding=True, truncation=True, return_tensors="pt")
val_targets = tokenizer(["Validation example 1", "Validation example 2"], padding=True, truncation=True, return_tensors="pt")

val_dataset = TensorDataset(val_inputs.input_ids, val_inputs.attention_mask, val_targets.input_ids)
val_dataloader = DataLoader(val_dataset, batch_size=2)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
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

        logging.info(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(dataloader)}, Validation Loss: {val_loss / len(val_dataloader)}")

# Cleanup
dist.destroy_process_group()