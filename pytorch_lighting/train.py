import json
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import torch
import yaml
from datasets import load_from_disk
from transformers import HfArgumentParser
from peft import LoraConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data import DataLoader

from llama_squad import LlamaSquadDataCollator

from model import (
    get_model_and_tokenizer,
)

from create_squad_dataset import config


@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    use_4bit: Optional[bool] = field(default=True)
    use_nested_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    num_train_epochs: Optional[int] = field(default=1)
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=False)
    packing: Optional[bool] = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: str = field(default="constant")
    lr_scheduler_kwargs: str = field(default="{}")
    max_steps: int = field(default=10000)
    eval_steps: int = field(default=1000)
    warmup_ratio: float = field(default=0)
    group_by_length: bool = field(default=True)
    save_steps: int = field(default=10)
    logging_steps: int = field(default=10)
    merge_and_push: Optional[bool] = field(default=False)
    output_dir: str = field(default="./results")
    apply_lora_to_all_layers: Optional[bool] = field(default=True)
    resume_from_checkpoint: Optional[str] = field(default=None)
    embedding_only: Optional[bool] = field(default=False)
    embedding_checkpoint: Optional[str] = field(default=None)
    node_count: int = field(
        default=1, metadata={"help": "Number of nodes for training"}
    )
    gpu_per_node: int = field(default=1, metadata={"help": "Number of GPUs per node"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    model, tokenizer, reasoning_tokens = get_model_and_tokenizer(
        model_name=config.model_name,
        quantize=args.use_4bit,
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    model.config.pretraining_tp = 1

    if args.apply_lora_to_all_layers:
        model_modules = str(model.modules)
        pattern = r"\\((\\w+)\\): Linear"
        linear_layer_names = re.findall(pattern, model_modules)
        target_modules = list(set(linear_layer_names))
    else:
        target_modules = None

    peft_config = LoraConfig(
        target_modules=target_modules,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return model, peft_config, tokenizer, reasoning_tokens


model, peft_config, tokenizer, reasoning_tokens = create_and_prepare_model(script_args)
model.config.use_cache = False
train_dataset = load_from_disk(config.dataset_name)["train"]
eval_dataset = load_from_disk(config.dataset_name)["val"]

tokenizer.padding_side = "right"

if "Llama-3" in tokenizer.name_or_path:
    answer_start_tokens = torch.tensor(
        tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\\n\\n",
            add_special_tokens=False,
        )
    )
    answer_end_tokens = torch.tensor(
        tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    )
else:
    answer_start_tokens = torch.tensor(
        tokenizer.encode("[/INST] ", add_special_tokens=False)
    )
    answer_end_tokens = torch.tensor(tokenizer.encode("</s>", add_special_tokens=False))


def tokenize_example(example):
    prompt_str = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    tokenized = tokenizer(
        prompt_str,
        max_length=script_args.max_seq_length,
        truncation=True,
        padding=False,
        return_tensors="np",
        add_special_tokens=False,
    )

    example["input_ids"] = tokenized["input_ids"][0].tolist()
    example["attention_mask"] = tokenized["attention_mask"][0].tolist()
    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
    }


train_dataset = train_dataset.map(tokenize_example, batched=False)
eval_dataset = eval_dataset.map(tokenize_example, batched=False)

keep_columns = ["input_ids", "attention_mask"]
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in keep_columns]
)
eval_dataset = eval_dataset.remove_columns(
    [col for col in eval_dataset.column_names if col not in keep_columns]
)

data_collator = LlamaSquadDataCollator(
    answer_start_tokens=answer_start_tokens,
    answer_end_tokens=torch.tensor([-100]),
    reasoning_tokens=reasoning_tokens,
    tokenizer=tokenizer,
    mlm=False,
)


class LlamaLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return optimizer


if script_args.embedding_only:
    for name, param in model.named_parameters():
        if "new_embedding" not in name:
            param.requires_grad = False

if script_args.resume_from_checkpoint or script_args.embedding_checkpoint:
    if script_args.embedding_checkpoint and hasattr(
        model.base_model.model.model.embed_tokens, "new_embedding"
    ):
        model.base_model.model.model.embed_tokens.new_embedding.weight = (
            torch.nn.Parameter(
                torch.load(
                    os.path.join(script_args.embedding_checkpoint, "embedding.pt"),
                    weights_only=True,
                ).to(
                    model.base_model.model.model.embed_tokens.new_embedding.weight.dtype
                )
            )
        )

train_dataloader = DataLoader(
    train_dataset,
    batch_size=script_args.per_device_train_batch_size,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=4,
)
val_dataloader = DataLoader(
    eval_dataset,
    batch_size=script_args.per_device_eval_batch_size,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=4,
)

lightning_module = LlamaLightningModule(model, tokenizer, script_args)

wandb_logger = WandbLogger(
    project="llama_squad",
    name=f"nodes_{script_args.node_count}_gpus_{script_args.gpu_per_node}",
)

trainer = pl.Trainer(
    default_root_dir=script_args.output_dir,
    max_epochs=script_args.num_train_epochs,
    gradient_clip_val=script_args.max_grad_norm,
    precision=16 if script_args.fp16 else 32,
    accumulate_grad_batches=script_args.gradient_accumulation_steps,
    val_check_interval=script_args.eval_steps if script_args.eval_steps > 0 else 1.0,
    log_every_n_steps=script_args.logging_steps,
    enable_checkpointing=True,
    logger=wandb_logger,
    devices=script_args.gpu_per_node,
    num_nodes=script_args.node_count,
    accelerator="gpu",
    gpus=script_args.gpu_per_node * script_args.node_count,
    strategy=DDPStrategy(find_unused_parameters=False),
)

trainer.fit(
    lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
os.makedirs(output_dir, exist_ok=True)
lightning_module.model.save_pretrained(output_dir)
if hasattr(lightning_module.model, "new_embedding"):
    torch.save(
        lightning_module.model.new_embedding.weight,
        os.path.join(output_dir, "embedding.pt"),
    )

if script_args.merge_and_push:
    from peft import AutoPeftModelForCausalLM

    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir, device_map="auto", torch_dtype=torch.bfloat16
    )
    merged_model = merged_model.merge_and_unload()
    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
    merged_model.save_pretrained(output_merged_dir, safe_serialization=True)
