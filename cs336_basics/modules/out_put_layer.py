import torch
import torch.nn as nn
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.rmsnorm import RMSNorm

class OutputLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        use_norm: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.norm = RMSNorm(d_model, device=device, dtype=dtype) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.linear(x)