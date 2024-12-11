import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import wandb
from datasets import load_dataset
import argparse
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


class LitLLM(LightningModule):
    def __init__(self, model_name, train_size, val_size, batch_size, eval_branch):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.eval_branch = eval_branch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.loss = torch.nn.CrossEntropyLoss()

    def prepare_data(self):
        dataset = load_dataset("squad_v2")
        self.train_data = dataset["train"].select(
            range(min(len(dataset["train"]), self.train_size))
        )
        self.val_data = dataset["validation"].select(
            range(min(len(dataset["validation"]), self.val_size))
        )

    def setup(self, stage=None):
        def preprocess_function(samples):
            questions = samples["question"]
            contexts = samples["context"]
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                questions, contexts, truncation=True, padding=True, max_length=512
            )
            labels = inputs.input_ids
            return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": labels,
            }

        self.train_data = self.train_data.map(
            preprocess_function,
            batched=True,
            remove_columns=self.train_data.column_names,
        )
        self.val_data = self.val_data.map(
            preprocess_function, batched=True, remove_columns=self.val_data.column_names
        )
        self.train_dataset = CustomDataset(self.train_data)
        self.val_dataset = CustomDataset(self.val_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss = self(input_ids, attention_mask, labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name"
    )
    parser.add_argument(
        "--train_size", type=int, default=500, help="Number of training examples"
    )
    parser.add_argument(
        "--val_size", type=int, default=50, help="Number of validation examples"
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="GPUs per node")
    parser.add_argument(
        "--eval_branch",
        type=int,
        default=1,
        choices=[0, 1],
        help="Evaluation branch: 0 (rank 0), 1 (all ranks)",
    )
    args = parser.parse_args()

    wandb_logger = WandbLogger(
        project="LLM_distri_training",
        name=f"{args.num_nodes} nodes, {args.gpus_per_node} gpus; eval: {args.eval_branch}",
    )

    model = LitLLM(
        model_name=args.model_name,
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        eval_branch=args.eval_branch,
    )

    trainer = pl.Trainer(
        max_epochs=3,
        devices=args.gpus_per_node,
        num_nodes=args.num_nodes,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=wandb_logger,
        precision=16,
    )

    trainer.fit(model)
