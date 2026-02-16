import torch
import torch.nn as nn
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.rope import RoPE

def stable_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    return exp_logits / sum_exp_logits

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == False, -torch.inf)
    attention = stable_softmax(scores, dim=-1)
    output = torch.matmul(attention, value)
    return output

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        use_rope: bool = False, 
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_linear = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
    
    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_in = x.shape
        causal_mask = self._causal_mask(seq_len, x.device)
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        if self.use_rope:
            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)
        attention = scaled_dot_product_attention(Q, K, V, causal_mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.o_linear(attention)