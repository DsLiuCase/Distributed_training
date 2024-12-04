import os
import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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
torch.cuda.set_device(int(gpu_ids[rank % len(gpu_ids)]))
device = torch.device("cuda")

# Initialize distributed environment
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=size,
    rank=rank,
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("huggingface/llama-1b")
model = AutoModelForCausalLM.from_pretrained("huggingface/llama-1b").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Prepare dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        return input_ids, attention_mask

texts = ["This is a training example.", "Another example for fine-tuning."]
train_dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for input_ids, attention_mask in dataloader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
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

    if rank == 0:
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

# Cleanup
dist.destroy_process_group()
dist.destroy_process_group()