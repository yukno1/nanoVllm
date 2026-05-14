import torch
from torch import nn
import torch.nn.functional as F
import time


class SiluAndMul(nn.Module):
    """
    A custom activation layer that applies the SiLU (sigmoid linear unit) activation funtion followed by selement-wise multiplication with the input tensor.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


if __name__ == "__main__":
    # example usage
    layer = SiluAndMul().cuda()
    input_tensor = torch.randn(8, 4000, 8000).cuda()

    # warm up iter
    for _ in range(10):
        _ = layer(input_tensor)

    times = []
    for i in range(100):
        print("run {0}", i)
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
