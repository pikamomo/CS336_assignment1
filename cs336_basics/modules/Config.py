from dataclasses import dataclass, fields, field
from typing import Any, Mapping
import json
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    max_seq_len: int = 256

    d_model: int = 512
    d_ff: int = 1344

    num_heads: int = 16
    num_layers: int = 4

    dropout: float = 0.1

    use_rms_norm: bool = True
    pre_norm: bool = True

    # Special token IDs
    eos_token_id: int = 256

    # RoPE
    use_rope: bool = True
    rope_theta: float = 10000.0

    # MoE specific parameters
    use_moe: bool = False
    num_experts: int = 4
    top_k: int = 1
    router_jitter: float = 0.1
    z_loss_coef: float = 1e-3
    lb_loss_coef: float = 1e-1

    # Others
    tie_weights: bool = False
    use_final_norm: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        allowed = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)



@dataclass
class TrainingConfig:
    batch_size: int = 256
    num_steps: int = 10_000
    dataset_dir: str = "datasets/tiny_stories"
    train_data_path: str = "datasets/tiny_stories/train.bin"
    eval_data_path: str = "datasets/tiny_stories/eval.bin"

    # Optimizer related parameters
    betas: tuple = field(default=(0.9, 0.98))
    weight_decay: float = 1e-5
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Logging & checkpointing
    wandb_logging: bool = True
    eval_log_interval: int = 500
    sampling_log_interval: int = 200

    # Others:
    model_name: str = "tiny_stories_transformer"
    save_checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    debug_mode: bool = False
    use_mixed_precision: bool = True
    seed: int = 2025