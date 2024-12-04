# import os
# import time
# import torch
# import torch.distributed as dist
# from mpi4py import MPI
# from torch.utils.data import DataLoader, TensorDataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import logging
# import wandb
# from datasets import load_dataset
# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")




# # Initialize wandb (only on rank 0)
# if MPI.COMM_WORLD.Get_rank() == 0:
#     wandb.init(project="llama-distributed", name="inference_latency_tracking")

# # Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Ensure MASTER_ADDR and MASTER_PORT are set
# os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
# os.environ.setdefault("MASTER_PORT", "12345")

# # Map rank to available GPUs
# gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
# device_id = int(gpu_ids[rank % len(gpu_ids)])
# torch.cuda.set_device(device_id)
# device = torch.device(f"cuda:{device_id}")

# # Initialize distributed environment
# dist.init_process_group(
#     backend="nccl",
#     init_method="env://",
#     world_size=size,
#     rank=rank,
# )




# # Load SQuAD 2.0 dataset
# dataset = load_dataset("squad_v2")

# # Split dataset
# train_dataset = dataset["train"]
# validation_dataset = dataset["validation"]


# # Load LLaMA model and tokenizer
# model_name = "meta-llama/Llama-3.2-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model = model.to(device)

# # Wrap model for distributed training
# model = torch.nn.parallel.DistributedDataParallel(
#     model, device_ids=[device_id], output_device=device_id
# )

# # Prepare dataset
# inputs = ["Hello, world!", "Fine-tuning LLaMA is fun!", "Distributed training is powerful."]
# targets = ["Hello, world!", "Fine-tuning LLaMA is fun!", "Distributed training is powerful."]

# # Tokenize data
# inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
# targets = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")

# dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, targets.input_ids)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # Validation set
# val_inputs = tokenizer(["Validation example 1", "Validation example 2"], padding=True, truncation=True, return_tensors="pt")
# val_targets = tokenizer(["Validation example 1", "Validation example 2"], padding=True, truncation=True, return_tensors="pt")

# val_dataset = TensorDataset(val_inputs.input_ids, val_inputs.attention_mask, val_targets.input_ids)
# val_dataloader = DataLoader(val_dataset, batch_size=2)

# # Optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# # Training loop
# epochs = 100
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0.0

#     for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
#         batch_input_ids = batch_input_ids.to(device)
#         batch_attention_mask = batch_attention_mask.to(device)
#         batch_labels = batch_labels.to(device)

#         # Forward pass
#         outputs = model(
#             input_ids=batch_input_ids,
#             attention_mask=batch_attention_mask,
#             labels=batch_labels,
#         )
#         loss = outputs.loss

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()

#         # Allreduce gradients
#         for param in model.parameters():
#             if param.grad is not None:
#                 dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
#                 param.grad.data /= size

#         optimizer.step()
#         epoch_loss += loss.item()

#     # Validation step (only rank 0 logs validation results)
#     if rank == 0:
#         model.eval()
#         val_loss = 0.0
#         inference_times = []  # Track inference times

#         with torch.no_grad():
#             for val_input_ids, val_attention_mask, val_labels in val_dataloader:
#                 val_input_ids = val_input_ids.to(device)
#                 val_attention_mask = val_attention_mask.to(device)
#                 val_labels = val_labels.to(device)

#                 # Measure inference time
#                 start_time = time.time()
#                 val_outputs = model(
#                     input_ids=val_input_ids,
#                     attention_mask=val_attention_mask,
#                     labels=val_labels,
#                 )
#                 elapsed_time = time.time() - start_time
#                 inference_times.append(elapsed_time)

#                 val_loss += val_outputs.loss.item()

#         avg_inference_time = sum(inference_times) / len(inference_times)
#         wandb.log({"Epoch": epoch + 1, "Training Loss": epoch_loss / len(dataloader),
#                    "Validation Loss": val_loss / len(val_dataloader), "Avg Inference Time (s)": avg_inference_time})

#         logging.info(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(dataloader)}, "
#                      f"Validation Loss: {val_loss / len(val_dataloader)}, "
#                      f"Avg Inference Time: {avg_inference_time:.4f}s")

# # Cleanup
# dist.destroy_process_group()
# if rank == 0:
#     wandb.finish()


import os
import time
import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import logging
import wandb
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def collate_fn(batch):
    # Convert list of dicts to dict of tensors
    input_ids = torch.stack([torch.tensor(sample["input_ids"]) for sample in batch])
    attention_mask = torch.stack([torch.tensor(sample["attention_mask"]) for sample in batch])
    start_positions = torch.tensor([sample["start_positions"] for sample in batch])
    end_positions = torch.tensor([sample["end_positions"] for sample in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_positions,
        "end_positions": end_positions,
    }

# Initialize wandb (only on rank 0)
if MPI.COMM_WORLD.Get_rank() == 0:
    wandb.init(project="squad2-distributed", name="fine-tuning")

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

# Load SQuAD 2.0 dataset
dataset = load_dataset("squad_v2")
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

# Load tokenizer and model for question-answering
model_name = "bert-base-uncased"  # Change to "meta-llama/Llama-3.2-1B" if required
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

# Wrap model for distributed training
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[device_id], output_device=device_id
)

# Preprocess function for SQuAD
def preprocess_data(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation=True,
        max_length=384,
        stride=128,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    offset_mapping = tokenized.pop("offset_mapping")
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers_sample = answers[sample_index]

        if len(answers_sample["text"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers_sample["answer_start"][0]
            end_char = start_char + len(answers_sample["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
validation_dataset = validation_dataset.map(preprocess_data, batched=True, remove_columns=validation_dataset.column_names)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(validation_dataset, batch_size=16, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch in train_dataloader:
        optimizer.zero_grad()

        # Move data to GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient Reduction
        for param in model.parameters():
            if param.grad is not None:
                comm.Allreduce(MPI.IN_PLACE, param.grad.data, op=MPI.SUM)
                param.grad.data /= size

        optimizer.step()
        epoch_loss += loss.item()

    # Validation step
    if rank == 0:
        model.eval()
        val_loss = 0.0
        inference_times = []

        with torch.no_grad():
            for batch in val_dataloader:
                batch_input_ids = batch["input_ids"].to(device)
                batch_attention_mask = batch["attention_mask"].to(device)
                batch_start_positions = batch["start_positions"].to(device)
                batch_end_positions = batch["end_positions"].to(device)

                start_time = time.time()
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    start_positions=batch_start_positions,
                    end_positions=batch_end_positions,
                )
                elapsed_time = time.time() - start_time
                inference_times.append(elapsed_time)

                val_loss += outputs.loss.item()

        avg_inference_time = sum(inference_times) / len(inference_times)
        wandb.log({"Epoch": epoch + 1, "Training Loss": epoch_loss / len(train_dataloader),
                   "Validation Loss": val_loss / len(val_dataloader),
                   "Avg Inference Time (s)": avg_inference_time})

        logging.info(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_dataloader)}, "
                     f"Validation Loss: {val_loss / len(val_dataloader)}, "
                     f"Avg Inference Time: {avg_inference_time:.4f}s")

# Cleanup
dist.destroy_process_group()
if rank == 0:
    wandb.finish()