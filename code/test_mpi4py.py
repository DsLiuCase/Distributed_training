import os
import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.utils.data import DataLoader, TensorDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure MASTER_ADDR and MASTER_PORT are set
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12345"

# Map rank to available GPUs
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

# print(f"Rank {rank}, Local GPU IDs: {gpu_ids}")
# if not gpu_ids or rank >= len(gpu_ids):
#     raise RuntimeError(f"Rank {rank} cannot find a valid GPU in CUDA_VISIBLE_DEVICES.")
torch.cuda.set_device(int(gpu_ids[rank % len(gpu_ids)]))
device = torch.device("cuda")

# Initialize distributed environment
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=size,
    rank=rank,
)

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize model and optimizer
model = SimpleModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create dummy dataset
inputs = torch.randn(1000, 10)
targets = torch.randn(1000, 1)
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create a validation set
val_inputs = torch.randn(200, 10)
val_targets = torch.randn(200, 1)
val_dataset = TensorDataset(val_inputs, val_targets)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch_inputs, batch_targets in dataloader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # Forward pass
        outputs = model(batch_inputs)
        loss = torch.nn.functional.mse_loss(outputs, batch_targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Allreduce gradients
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size  # Average gradients across processes

        optimizer.step()
        epoch_loss += loss.item()

    # Validation step (only rank 0 logs validation results)
    if rank == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch_inputs, val_batch_targets in val_dataloader:
                val_batch_inputs, val_batch_targets = val_batch_inputs.to(device), val_batch_targets.to(device)
                val_outputs = model(val_batch_inputs)
                val_loss += torch.nn.functional.mse_loss(val_outputs, val_batch_targets).item()

        logging.info(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(dataloader)}, Validation Loss: {val_loss / len(val_dataloader)}")

# Cleanup
dist.destroy_process_group()