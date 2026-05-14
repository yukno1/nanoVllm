import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any, Callable

from utils import get_context


class Parameter(nn.Parameter):
    weight_loader: Callable


# vocabparallelembedding
# shard over the number of vocab, not the embedding size


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        self.num_embeddings = num_embeddings
        # pad to make it divisible by tp_size
        self.padded_num_embeddings = (
            (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        )
        # this is the num_embeddings per partition in this current GPU
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim

        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mask for tokens in this partition's range and within original vocab size
        mask = (
            (x >= self.vocab_start_idx)
            & (x < self.vocab_end_idx)
            & (x < self.num_embeddings)
        )
        x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # need to mask again, otherwise the embedding for the out-of-range ids will be the embedding of id 0
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__(num_embeddings, embedding_dim)

    # x: [batch_size, seq_len, hidden_size]
    # weight: [vocab_size_per_partition, hidden_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            assert context.cu_seqlens_q is not None, "context.cu_seqlens_q must be set."
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # logits: [batch_size, seq_len, vocab_size_per_partition]
        # F.linear automatically transpose the weight
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # prepare for all_gather only for GPU 0 which is the main GPU
            all_logits = (
                [
                    torch.empty(logits.size(), device=logits.device)
                    for _ in range(self.tp_size)
                ]
                if self.tp_rank == 0
                else None
            )
            # dist.gather collects the logits from all GPUs to GPU 0
            dist.gather(logits, gather_list=all_logits, dst=0)
            # concatenate
            if self.tp_rank == 0:
                # [batch_size, seq_len, padded_vocab_size]
                logits = torch.cat(all_logits, dim=-1)
                # trim to original vocab size
                logits = logits[..., : self.num_embeddings]

        return logits
