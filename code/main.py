import os
import time
import torch
from mpi4py import MPI
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForQuestionAnswering

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

os.environ["LOCAL_RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)

# 配置 GPU 环境
local_rank = rank % torch.cuda.device_count()
torch.cuda.set_device(local_rank)

def preprocess_function(examples, tokenizer):
    """数据预处理函数"""
    inputs = examples["question"]
    targets = examples["context"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # 加载模型和数据
    model = LlamaForQuestionAnswering.from_pretrained("Llama3-base").cuda(local_rank)
    tokenizer = LlamaTokenizer.from_pretrained("Llama3-base")
    dataset = load_dataset("squad_v2")

    # 数据预处理
    tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    train_dataset = tokenized_datasets["train"]

    # 将数据集分割为多个部分，每个节点处理一个部分
    data_per_rank = len(train_dataset) // world_size
    start_idx = rank * data_per_rank
    end_idx = (rank + 1) * data_per_rank if rank != world_size - 1 else len(train_dataset)
    local_dataset = train_dataset.select(range(start_idx, end_idx))

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 模拟多 epoch 训练
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()

        for step, batch in enumerate(local_dataset):
            # 将数据移动到 GPU
            input_ids = torch.tensor(batch["input_ids"]).cuda(local_rank)
            attention_mask = torch.tensor(batch["attention_mask"]).cuda(local_rank)
            labels = torch.tensor(batch["labels"]).cuda(local_rank)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # 反向传播计算本地梯度
            optimizer.zero_grad()
            loss.backward()

            # 聚合所有节点的梯度
            for param in model.parameters():
                if param.grad is not None:
                    grad_tensor = param.grad.data
                    reduced_grad = torch.zeros_like(grad_tensor)
                    comm.Allreduce(grad_tensor, reduced_grad, op=MPI.SUM)
                    param.grad.data = reduced_grad / world_size  # 计算平均梯度

            # 更新模型参数
            optimizer.step()

        # 记录当前 epoch 的耗时
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # 通过 rank-0 输出 epoch 时间
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f} seconds")

    # 保存模型（仅 rank-0 保存）
    if rank == 0:
        model.save_pretrained("./finetuned-llama3")
        tokenizer.save_pretrained("./finetuned-llama3")

if __name__ == "__main__":
    main()