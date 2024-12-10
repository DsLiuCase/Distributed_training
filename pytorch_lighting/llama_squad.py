from typing import Optional

import torch
from transformers import DataCollatorForLanguageModeling, LlamaConfig, LlamaForCausalLM


class LlamaSquadDataCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        answer_start_tokens: torch.Tensor,
        answer_end_tokens: torch.Tensor,
        reasoning_tokens: Optional[torch.Tensor],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.answer_start_tokens = answer_start_tokens
        self.answer_end_tokens = answer_end_tokens
        self.reasoning_tokens = reasoning_tokens

    def __call__(self, examples):
        batch = super().__call__(examples)

        for i, label in enumerate(batch["labels"]):
            # Only apply cross entropy loss to the answer part of the labels
            mask = torch.ones_like(label)
            window = label.unfold(0, self.answer_start_tokens.shape[0], 1)
            answer_starts = (window == self.answer_start_tokens).all(dim=1).nonzero()[
                :, 0
            ] + self.answer_start_tokens.shape[0]
            window = label.unfold(0, self.answer_end_tokens.shape[0], 1)
            answer_ends = (window == self.answer_end_tokens).all(dim=1).nonzero()[
                :, 0
            ] + self.answer_end_tokens.shape[0]
            for answer_start in answer_starts:
                mask[answer_start : answer_ends[answer_ends > answer_start][0]] = 0
            label = label.where(mask == 0, -100)

            # Mask out the reasoning tokens
            if self.reasoning_tokens is not None:
                mask = (label.unsqueeze(1) == self.reasoning_tokens).any(dim=1)
                label = torch.where(mask, torch.tensor(-100), label)

            batch["labels"][i] = label

        return batch


class ExtendedEmbedding(torch.nn.Module):
    def __init__(
        self, original_embedding: torch.nn.Embedding, new_embedding: torch.nn.Embedding
    ):
        super(ExtendedEmbedding, self).__init__()
        self.original_embedding = original_embedding
        self.new_embedding = new_embedding

    def forward(self, input_ids):
        # Determine the device and dtype from the input_ids or from original_embedding
        device = input_ids.device
        # original_embeddings will be on the same device as original_embedding
        is_new_token = input_ids >= self.original_embedding.num_embeddings
        original_tokens = input_ids[~is_new_token]
        original_embeddings = self.original_embedding(original_tokens)

        # Create the combined_embeddings on the input_ids device and correct dtype
        combined_embeddings = torch.zeros(
            input_ids.shape + (original_embeddings.shape[1],),
            device=device,
            dtype=original_embeddings.dtype,
        )
        combined_embeddings[~is_new_token] = original_embeddings

        new_tokens = input_ids[is_new_token] - self.original_embedding.num_embeddings
        if len(new_tokens) > 0:
            # The new_embedding is already on the correct device due to Lightning setup
            combined_embeddings[is_new_token] = self.new_embedding(new_tokens)

        return combined_embeddings


class LlamaSquadModel(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, num_new_tokens: int):
        super().__init__(config)
        if num_new_tokens > 0:
            self.new_embedding = torch.nn.Embedding(
                num_embeddings=num_new_tokens, embedding_dim=config.hidden_size
            )

    def patch_embeddings(self):
        if hasattr(self, "new_embedding"):
            self.base_model.embed_tokens = ExtendedEmbedding(
                self.base_model.embed_tokens, self.new_embedding
            )
