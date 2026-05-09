from typing import Any

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self):
        pass
