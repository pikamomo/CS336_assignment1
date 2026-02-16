import torch
import torch.nn as nn
from cs336_basics.modules.Config import ModelConfig
from cs336_basics.modules.embedding import Embedding
from cs336_basics.modules.transformer_block import TransformerBlock
from cs336_basics.modules.out_put_layer import OutputLayer
from cs336_basics.modules.rmsnorm import RMSNorm

class TransformerLM(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config
        self.token_embedding = Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model) 
        self.output_layer = OutputLayer(config.d_model, config.vocab_size, use_norm=config.use_final_norm)

        if config.tie_weights:
            self._tie_weights()
    
    def _tie_weights(self):
        self.output_layer.linear.weight = self.token_embedding.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.output_layer(x)