import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable


# 模拟分布式环境
class Dist:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self._rank_pool = set(range(world_size))

    def get_rank(self):
        if not self._rank_pool:
            raise RuntimeError("No avaliable rank left")
        r = self._rank_pool.pop()
        print(rf"rank:{r}")
        return r

    def get_world_size(self):
        return self.world_size


dist = Dist(world_size=4)


class Parameter(nn.Parameter):
    weight_loader: Callable


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
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


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size()
        assert (
            output_size % tp_size == 0
        ), "Output size must be divisible by tensor parallel size."
        super().__init__(
            input_size, output_size // tp_size, bias, tp_dim=0
        )  # tp_dim=0 -> column parallel

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        """
        :param
        """
        assert self.tp_dim is not None, "tp_dim must be set for ColumnParallelLinear"
        param_data = param.data
        full_data_output_size = loaded_weight.size(0)
        # output size of each gpu
        shard_size = (
            full_data_output_size // self.tp_size
        )  # param.data.size(self.tp_dim)
        assert shard_size == param.data.size(
            0
        ), "Shard size dows not match the parameter size."
        start_idx = self.tp_rank * shard_size  # offset
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


if __name__ == "__main__":
    input_size = 4
    output_size = 12
    world_size = dist.get_world_size()

    # 定义权重
    shard_out = output_size // world_size
    shards = []
    for rank in range(world_size):
        shard = torch.full(
            (shard_out, input_size), fill_value=rank + 1, dtype=torch.float32
        )
        shards.append(shard)
    full_weight = torch.cat(shards, dim=0).cuda()
    print(f"full_weight.shape = {full_weight.shape}")

    # print weight
    def print_weight(layer):
        print(f"Before loading weight:")
        print(layer.weight)
        layer.weight_loader(layer.weight, full_weight)
        print("After loading weight:")
        print(layer.weight)
        print()

    for _ in range(world_size):
        layer = ColumnParallelLinear(input_size, output_size)
        print_weight(layer)

"""
full_weight.shape = torch.Size([12, 4])
rank:0
Before loading weight:
Parameter containing:
Parameter(Parameter([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], requires_grad=True))
After loading weight:
Parameter containing:
Parameter(Parameter([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]], requires_grad=True))

rank:1
Before loading weight:
Parameter containing:
Parameter(Parameter([[2.5707e-20, 0.0000e+00, 0.0000e+00, 0.0000e+00],
           [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00],
           [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00]],
          requires_grad=True))
After loading weight:
Parameter containing:
Parameter(Parameter([[2., 2., 2., 2.],
           [2., 2., 2., 2.],
           [2., 2., 2., 2.]], requires_grad=True))

rank:2
Before loading weight:
Parameter containing:
Parameter(Parameter([[2.5687e-20, 0.0000e+00, 0.0000e+00, 0.0000e+00],
           [2.0000e+00, 2.0000e+00, 2.0000e+00, 2.0000e+00],
           [2.0000e+00, 2.0000e+00, 2.0000e+00, 2.0000e+00]],
          requires_grad=True))
After loading weight:
Parameter containing:
Parameter(Parameter([[3., 3., 3., 3.],
           [3., 3., 3., 3.],
           [3., 3., 3., 3.]], requires_grad=True))

rank:3
Before loading weight:
Parameter containing:
Parameter(Parameter([[2.5709e-20, 0.0000e+00, 0.0000e+00, 0.0000e+00],
           [3.0000e+00, 3.0000e+00, 3.0000e+00, 3.0000e+00],
           [3.0000e+00, 3.0000e+00, 3.0000e+00, 3.0000e+00]],
          requires_grad=True))
After loading weight:
Parameter containing:
Parameter(Parameter([[4., 4., 4., 4.],
           [4., 4., 4., 4.],
           [4., 4., 4., 4.]], requires_grad=True))
"""
