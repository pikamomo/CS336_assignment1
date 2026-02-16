import torch.nn as nn
import cs336_basics.modules.attention as attention
import cs336_basics.modules.ffn as ffn
import cs336_basics.modules.rmsnorm as rmsnorm
from cs336_basics.modules.Config import ModelConfig
import torch

class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config

        self.mha = attention.MultiheadAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            use_rope=config.use_rope,
        )

        self.ffn = ffn.FFN(
            d_model=config.d_model,
            d_ff=config.d_ff,
        )
        self.norm1 = rmsnorm.RMSNorm(d_model=config.d_model)
        self.norm2 = rmsnorm.RMSNorm(d_model=config.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.mha(self.norm1(x), token_positions=token_positions)
        x = x + self.ffn(self.norm2(x))
        return x