import os
from datetime import datetime
import torch
import time
import torch.distributed as dist
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import wandb
from datasets import load_dataset
import argparse
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
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

class main():
  def __init__(self, num_gpu_per_node, num_nodes, eval_branch):
    self.comm = MPI.COMM_WORLD
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=self.size,
        rank=self.rank,
    )
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    print("gpu_ids = ", gpu_ids)
    device_id = int(gpu_ids[self.rank % len(gpu_ids)])
    print("device_id = ", device_id)
    torch.cuda.set_device(device_id)
    self.device = torch.device(f"cuda:{device_id}")
    logging.info(f"rank = {self.rank}, size = {self.size}, PID = {os.getpid()}, gpus in current rank sets = {gpu_ids}, device_id = {device_id}")
    
    self.eval = eval_branch
    if MPI.COMM_WORLD.Get_rank() == 0:
        text = 'inference in rank0' if eval_branch == 0 else 'inference in all ranks' 
        wandb.init(project="LLM_distri_training", name=f"{num_nodes} nodes, {num_gpu_per_node} gpu in each node; {text}", mode="online")
        
    self.model, self.tokenizer = self.get_model() 

    # Load a subset of the SQuAD 2.0 dataset
    self.NUM_TRAIN_EXAMPLES = 500  # Define the number of training examples to use
    self.NUM_VAL_EXAMPLES = 50  # Define the number of validation examples to use
    self.global_train_size = -1
    self.global_train_size = -1
    self.train_dataloader, self.val_dataloader = self.get_data_loader(batch_size = 10)
    
  def get_model(self):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    model = model.to(self.device)
    logging.info("Model loaded successfully")
    return model, tokenizer
  
  def get_data_loader(self, batch_size = 1):
    dataset = load_dataset("squad_v2")
    train_data = dataset["train"].select(range(min(len(dataset["train"]), self.NUM_TRAIN_EXAMPLES)))
    validation_data = dataset["validation"].select(range(min(len(dataset["validation"]), self.NUM_VAL_EXAMPLES)))
    # Preprocess datasets
    train_data = train_data.map(self.preprocess_function, batched=True, remove_columns=train_data.column_names)
    validation_data = validation_data.map(self.preprocess_function, batched=True, remove_columns=validation_data.column_names)
    
    # Convert processed data into custom datasets
    train_dataset = CustomDataset(train_data)
    validation_dataset = CustomDataset(validation_data)

    # Create DataLoader with DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, pin_memory=True)

    val_sampler = DistributedSampler(validation_dataset)
    val_dataloader = DataLoader(validation_dataset, sampler=val_sampler, batch_size=batch_size, pin_memory=True)
    
    logging.info("Dataset loaded successfully")
    logging.info(f"Size: train_dataset: {len(train_dataset)}, val_dataset: {len(validation_dataset)} in rank {self.rank} with size {self.size}")
    size_of_train = len(train_dataset)
    size_of_eval = len(validation_dataset)
    self.global_train_size = size_of_train
    self.global_eval_size = size_of_eval
    return train_dataloader, val_dataloader

  # Preprocessing function
  def preprocess_function(self, samples):
      questions = samples["question"]
      contexts = samples["context"]
      if self.tokenizer.pad_token is None:  # If no padding token is set
          self.tokenizer.pad_token = self.tokenizer.eos_token  # Set eos_token as the padding token
      inputs = self.tokenizer(questions, contexts, truncation=True, padding=True, max_length=512)
      labels = inputs.input_ids
      return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": labels}

  def training(self, num_epochs=3, reduce_method=0):
      """
      reduce_method: 
          0: Use rank 0 for gradient aggregation and parameter updates
          1: Use all ranks to reduce gradients (Allreduce)
      """
      optimizer = AdamW(self.model.parameters(), lr=5e-5)
      for epoch in range(num_epochs):
          self.model.train()
          epoch_start_time = time.time()
          logging.info(f"\nEpoch {epoch} started, in rank {self.rank}, with local dataset size: {len(self.train_dataloader)}, with batch size: {self.train_dataloader.batch_size}") 
          for batch in self.train_dataloader:
              batch_start_time = time.time()
              
              # Load batch to device
              input_ids, attention_mask, labels = batch
              input_ids, attention_mask, labels = (
                  input_ids.to(self.device),
                  attention_mask.to(self.device),
                  labels.to(self.device),
              )
              
              # Forward pass
              outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
              loss = outputs.loss
              loss.backward()
              
              # Gradient synchronization and parameter updates
              if reduce_method == 0:  # Reduce + Broadcast
                  for param in self.model.parameters():
                      local_grad = param.grad.clone()
                      global_grad = torch.zeros_like(local_grad)
                      self.comm.Reduce(local_grad.cpu().numpy(), global_grad.cpu().numpy(), op=MPI.SUM, root=0)
                      
                      if self.rank == 0:  # Only rank 0 updates parameters
                          param.grad.copy_(global_grad / self.size)
                  if self.rank == 0:
                      optimizer.step()
                      optimizer.zero_grad()
                  for param in self.model.parameters():
                      self.comm.Bcast(param.data.cpu().numpy(), root=0)
                      
              elif reduce_method == 1:  # Allreduce
                  for param in self.model.parameters():
                      local_grad = param.grad.clone()
                      global_grad = torch.zeros_like(local_grad)
                      self.comm.Allreduce(local_grad.cpu().numpy(), global_grad.cpu().numpy(), op=MPI.SUM)
                      param.grad.copy_(global_grad / self.size)
                  optimizer.step()
                  optimizer.zero_grad()
              
              # Log batch update time
              batch_end_time = time.time()
              if self.rank == 0:
                  batch_update_time = batch_end_time - batch_start_time
                  logging.info(f"Batch eclipse time in rank 0: {batch_update_time}")
                  wandb.log({"training_batch_eclipse": batch_update_time})
          
          # End of epoch logging
          if self.rank == 0:
              epoch_end_time = time.time()
              epoch_time = epoch_end_time - epoch_start_time
              logging.info(f"Epoch {epoch} completed, loss: {loss.item()}, time: {epoch_time}")
              wandb.log({"loss": loss.item(), "training_epoch_eclipse": epoch_time})
          
          self.evaluation() 
    

  def evaluation(self, eval_method=0):
      """
      eval_method:
          0: Only rank 0 performs evaluation on the entire dataset (single-process evaluation).
          1: All ranks perform evaluation on the entire dataset (multi-process evaluation).
      """
      self.model.eval()
      val_loss = 0.0  # Accumulate loss
      total_inference_time = 0.0  # Track inference time

      if eval_method == 0:
          # Branch 0: Single-process evaluation on rank 0
          if self.rank == 0:
              start_time = time.time()  # Start timing
              with torch.no_grad():
                  for val_input_ids, val_attention_mask, val_labels in self.val_dataloader:
                      # Move data to device
                      val_input_ids = val_input_ids.to(self.device)
                      val_attention_mask = val_attention_mask.to(self.device)
                      val_labels = val_labels.to(self.device)

                      # Forward pass
                      val_outputs = self.model(
                          input_ids=val_input_ids,
                          attention_mask=val_attention_mask,
                          labels=val_labels,
                      )
                      val_loss += val_outputs.loss.item()

              total_inference_time = time.time() - start_time  # End timing
              per_inference_time = total_inference_time / len(self.val_dataloader)
              total_inference_time = per_inference_time * self.global_eval_size

              # Compute average validation loss
              avg_val_loss = val_loss / len(self.val_dataloader)

              # Return single-process results
              wandb.log({"avg_inference_loss": avg_val_loss, \
                         "total_inference_time": total_inference_time, \
                         "per_inference_time": per_inference_time})  
              logging.info(f"Avg. Val Loss: {avg_val_loss}, Total Inference Time: {total_inference_time}, \
                            per_inference_time: {per_inference_time}, \
                           inference in rank 0 only")
              return avg_val_loss, total_inference_time
          else:
              # Other ranks do nothing
              return None, None

      elif eval_method == 1:
          # Branch 1: Multi-process evaluation
          start_time = time.time()  # Start timing
          with torch.no_grad():
              for val_input_ids, val_attention_mask, val_labels in self.val_dataloader:
                  # Move data to device
                  val_input_ids = val_input_ids.to(self.device)
                  val_attention_mask = val_attention_mask.to(self.device)
                  val_labels = val_labels.to(self.device)

                  # Forward pass
                  val_outputs = self.model(
                      input_ids=val_input_ids,
                      attention_mask=val_attention_mask,
                      labels=val_labels,
                  )
                  val_loss += val_outputs.loss.item()

          local_inference_time = time.time() - start_time  # End timing

          # Aggregate results across all ranks
          val_loss_tensor = torch.tensor(val_loss).to(self.device)
          local_inference_time_tensor = torch.tensor(local_inference_time).to(self.device)

          # Synchronize loss and time across all ranks
          self.comm.Allreduce(MPI.IN_PLACE, val_loss_tensor, op=MPI.SUM)
          self.comm.Allreduce(MPI.IN_PLACE, local_inference_time_tensor, op=MPI.MAX)
          per_inference_time_tensor /= len(self.val_dataloader)  # Average inference time across all ranks
          total_inference_time_tensor = per_inference_time_tensor * self.global_eval_size
          # Compute global average loss
          avg_val_loss = val_loss_tensor.item() / self.size

          # Return multi-process results
          wandb.log({"avg_val_loss": avg_val_loss, "total_inference_time": local_inference_time_tensor.item(), \
                     "per_inference_time": per_inference_time_tensor.item()})
          logging.info(f"Avg. Val Loss: {avg_val_loss}, Total Inference Time: {local_inference_time_tensor.item()}, \
                        per_inference_time: {per_inference_time_tensor.item()}, \
                       inference in all ranks")
          return avg_val_loss, local_inference_time_tensor.item()
      else:
          return None, None


  def run(self):
    self.training()
    print("completed in the time:", datetime.now())
    if self.rank == 0:
        wandb.finish()
        
    
if __name__ == "__main__":
    # 解析命令行参数
    print("start in the time:", datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpu_per_node", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--eval_branch", "-EB", choices=[0, 1], type=int, default=1, help="Evaluation method: 1 for all ranks, 0 for only rank 0")
    args = parser.parse_args()
    logging.info(f"Number of GPUs per node: {args.num_gpu_per_node}, Number of nodes: {args.num_nodes}, Evaluation branch: {args.eval_branch}")
    main = main(args.num_gpu_per_node, args.num_nodes, args.eval_branch)
    main.run()
    # # 设置分布式环境
    # setup_distributed()

    # # 每个进程的编号
    # # rank = dist.get_rank()
    # # # 进程总数
    # # world_size = dist.get_world_size()

    # print(f"rank: {rank}, world_size: {world_size}")
    print("completed in the time:", datetime.now())