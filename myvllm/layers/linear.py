import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Callable, Literal


class Parameter(nn.Parameter):
    weight_loader: Callable


class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: Literal[0, 1, None] = None,
    ) -> None:
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # create weight parameter with custom weight loader
        self.weight = Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        raise NotImplementedError("Subclass should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass should implement this method.")


"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""


class ReplicatedLinear(LinearBase):
    """
    最简单的线性层: ReplicatedLinear (继承自 LinearBase)

    weight_loader 就是简单地把权重拷贝过来（不做任何切分）

    forward 就是一个普通线性层的计算
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias, 0)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层 (Column Parallel Linear)。

    用于张量并行 (TP) 场景：把完整的输出维度切分成 tp_size 份，
    每张 GPU 只负责其中一份。即权重矩阵沿"输出维" (dim 0)切分，
    属于 tp_dim=0 的切分方式。

    切分示意 (tp_size=2):
        完整权重 W [output, input]
        ┌──────────────────┐
        │ GPU0: 前一半行   │  → 本卡权重形状 [output/2, input]
        │ GPU1: 后一半行   │
        └──────────────────┘

    由于各卡输出的行互不重叠，前向计算后【不需要】通信
    （与 RowParallelLinear 的 all_reduce 不同）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size()
        # 完整输出维必须能被 GPU 数量整除，否则无法均匀切分
        assert output_size % tp_size == 0, (
            "Output size must be divisible by tensor parallel size."
        )
        # 每张卡只持有 output_size // tp_size 行的权重
        # tp_dim=0 表示沿权重W的输出维（第 0 维）切分
        super().__init__(
            input_size, output_size // tp_size, bias, tp_dim=0
        )  # tp_dim=0 -> column parallel

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        """
        把 checkpoint 中的完整权重加载到当前 GPU 的分片上。

        :param param: 当前卡上待填充的参数，形状 [output/tp_size, input]
        :param loaded_weight: checkpoint 中的完整权重，形状 [output, input]
        """
        # assert self.tp_dim is not None, "tp_dim must be set for ColumnParallelLinear"
        param_data = param.data
        full_data_output_size = loaded_weight.size(0)
        # output size of each gpu
        shard_size = (
            full_data_output_size // self.tp_size
        )  # param.data.size(self.tp_dim)
        assert shard_size == param.data.size(0), (
            "Shard size dows not match the parameter size."
        )
        start_idx = self.tp_rank * shard_size  # offset
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        *,
        loaded_shard_id: int = -1,
    ) -> None:
        # assert self.tp_dim is not None, (
        #     "tp_dim must be set for MergedColumnParallelLinear"
        # )
        assert self.loaded_shard_id == -1, (
            "loaded_shard_id must be set for MergedColumnParallelLinear"
        )
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        assert total_num_heads % tp_size == 0, (
            "total_num_heads must be multiples of tp_size"
        )
        self.num_heads = total_num_heads // tp_size
        assert total_num_kv_heads % tp_size == 0, (
            "total_num_kv_heads must be multiples of tp_size"
        )
        self.num_kv_heads = total_num_kv_heads // tp_size
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        *,
        loaded_shard_id: str = "",
    ) -> None:
        assert loaded_shard_id in [
            "q",
            "k",
            "v",
        ], "loaded_shard_id must be 'q', 'k', or 'v'."
        # assert self.tp_dim is not None, (
        #     "tp_dim must be set for MergedColumnParallelLinear"
        # )
        # batch_size * num_heads * num_token * head_size
        param_data = param.data

        # loaded_weights: batch_size * num_token * (head_size*num_heads)
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # shard the original full weight
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, (
            "Input size must be divisible by tensor parallel size."
        )
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        # assert self.tp_dim is not None, (
        #     "tp_dim must be set for MergedColumnParallelLinear"
        # )

        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        slided_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y


if __name__ == "__main__":
    # Example usage
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    layer = LinearBase(input_size=10, output_size=5)
    print("LinearBase layer initialized:", layer)
