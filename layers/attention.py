from typing import Any

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

    def forward(self):
        pass
