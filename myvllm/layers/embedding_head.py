import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Callable

from ..utils import get_context


class Parameter(nn.Parameter):
    weight_loader: Callable


class VocabParallelEmbedding(nn.Module):
    """
    VocabParallelEmbedding
    词表维度的张量并行 (TP) 嵌入层。

    与普通线性层按 hidden_size 切分不同，词嵌入层是按 vocab 维度切分：
      - 每个 GPU 只持有 [num_embeddings / tp_size, embedding_dim] 的权重切片
      - 每个 token id 只会落在某一个 GPU 的负责区间内
      - 各 GPU 对属于自己的 token 查表，其余位置填 0
      - 最后 all_reduce 求和，得到完整的嵌入向量
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        # 张量并行相关：总卡数 + 当前卡号
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # 保存原始词表大小 (padding 之前的大小，加载和裁剪时会用到)
        self.num_embeddings = num_embeddings
        # 向上取整到 tp_size 的倍数，保证每个分片长度相等
        self.padded_num_embeddings = (
            (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        )
        # 当前 GPU 上实际持有的词表大小
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim

        # 本分片负责的 token id 区间: [vocab_start_idx, vocab_end_idx)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 本卡只分配自己那一份权重: [num_embeddings_per_partition, embedding_dim]
        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        # 绑定自定义加载器：加载时按 rank 从完整权重中切出本卡的分片
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """
        从 checkpoint 的完整权重 [num_embeddings, embedding_dim] 中，
        沿 dim=0 切出本 rank 负责的那一段，拷贝到本地参数。
        """

        param_data = param.data
        shard_size = param_data.size(0)  # 本卡分片大小
        start_idx = self.tp_rank * shard_size  # 本卡在完整权重中的起始行
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:  token id 张量, 任意形状
        输出:  嵌入向量, 形状 = x.shape + [embedding_dim]

        步骤:
          1. 计算 mask, 标记哪些 token 属于本分片区间
             （同时剔除 padding 出来的、超过原始 vocab 大小的 id)
          2. 把本分片内的 token 映射为本地索引 [0, per_partition),
             不属于本片的 id 先变成 0 (查 weight[0] 占位)
          3. 查表得到嵌入
          4. TP>1 时把非本片的嵌入清零（避免 weight[0] 的脏数据），
             再 all_reduce 求和，各卡的非零部分互不重叠
        """
        # mask 同时覆盖三个条件：
        #   - id >= 本卡起始位置
        #   - id < 本卡结束位置
        #   - id < 原始词表大小（排除 padding 出来的 id）
        # mask for tokens in this partition's range and within original vocab size
        mask = (
            (x >= self.vocab_start_idx)
            & (x < self.vocab_end_idx)
            & (x < self.num_embeddings)
        )
        # mask 中 False 的位置会被乘成 0，因此先查 weight[0] 也无妨
        x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # 必须再次掩码：否则超出本片范围的 id 会拿到 weight[0] 的嵌入 (即 id 0 的向量)，而不是 0 向量
            y = mask.unsqueeze(1) * y
            # 各卡的非零嵌入互不重叠，求和即可拼出完整的嵌入向量
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    LM Head 的并行实现，继承 VocabParallelEmbedding 复用其分片逻辑。

    与 embedding 的差异：
      - embedding 需要 all_reduce (每张卡都要拿到完整嵌入供下游计算)
      - LM head 的 logits 只有采样端需要，因此用 dist.gather 只汇总到 rank 0
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__(num_embeddings, embedding_dim)

    # x: [batch_size, seq_len, hidden_size]
    # weight: [vocab_size_per_partition, hidden_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            # prefill 阶段序列是变长的、拼接成一维输入的，每个序列只需要最后一个位置的 logits 用于采样
            assert context.cu_seqlens_q is not None, "context.cu_seqlens_q must be set."
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # logits: [batch_size, seq_len, vocab_size_per_partition]
        # F.linear 会自动转置 weight
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 只为 GPU 0 分配接收缓冲区（其余 rank 传 None）
            all_logits = (
                [
                    torch.empty(logits.size(), device=logits.device)
                    for _ in range(self.tp_size)
                ]
                if self.tp_rank == 0
                else None
            )
            # dist.gather 把各 GPU 的 logits 分片收集到 rank 0
            dist.gather(logits, gather_list=all_logits, dst=0)
            # 在 rank 0 上拼接 + 裁剪回原始词表大小
            if self.tp_rank == 0:
                # [batch_size, seq_len, padded_vocab_size]
                logits = torch.cat(all_logits, dim=-1)
                # 去掉 padding 的部分，恢复为 [batch_size, seq_len, num_embeddings]
                logits = logits[..., : self.num_embeddings]

        return logits
