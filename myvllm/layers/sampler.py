import torch
from torch import nn


class SamplerLayer(nn.Module):
    """
    A custom sampler layer that selects elements from the input tensor
    based on provided indices.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=-1))
        probs = torch.softmax(logits, dim=-1)
        sample_token = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_token
