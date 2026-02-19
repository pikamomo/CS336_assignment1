import torch
import numpy as np
import os
import wandb
from cs336_basics.modules.Config import ModelConfig
from cs336_basics.modules.Config import TrainingConfig

from cs336_basics.modules.Config import get_ctx
from cs336_basics.tokenizer import load_tokenizer_from_dir
from cs336_basics.modules.Config import data_loading_sequential
from cs336_basics.loss import cross_entropy
from cs336_basics.loss import perplexity
from cs336_basics.AdamW import gradient_clip
from cs336_basics.AdamW import cosine_annealing_lr
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.dataloader import data_loading_sequential
import gc
from cs336_basics.generate import generate
from tqdm import trange
from contextlib import nullcontext
from cs336_basics.dataloader import BatchState

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_config: TrainingConfig):
    tokenizer = load_tokenizer_from_dir(train_config.dataset_dir)

    # Load training dataset
    original_data = np.memmap(
        train_config.train_data_path,
        dtype=np.uint16,
        mode="r+",
    )
    x = torch.from_numpy(original_data)

    best_eval_loss = float("inf")
    ctx = get_ctx(train_config.use_mixed_precision, train_config.device)

    # Training loop
    state = BatchState(pos=0)
    for step in range(train_config.num_steps):
        log_dict = {}

        inputs, targets = data_loading_sequential(
            x=x,
            batch_size=train_config.batch_size,
            context_length=model.config.max_seq_len,
            device=train_config.device,
            state=state,
        )

        # Forward pass
        with ctx:
            logits, aux = model(inputs)

            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = cross_entropy(logits, targets)

            if model.config.use_moe:
                # Scale z-loss
                z_loss_scaled = aux["z_loss_scaled"]
                moe_layers = aux["moe_layers"]
                loss = loss + (z_loss_scaled / moe_layers)

                lb_loss = aux["lb_loss_scaled"]
                loss = loss + (lb_loss / moe_layers)

        # Backward pass and optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping
        gradient_clip(model.parameters(), max_l2_norm=train_config.max_grad_norm)

        # Learning rate scheduling
        lr = cosine_annealing_lr(
            t=step,
            alpha_max=train_config.max_lr,
            alpha_min=train_config.min_lr,
            Tw=train_config.warmup_steps,
            Tc=train_config.num_steps - train_config.warmup_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        # Logging
        if train_config.wandb_logging:
            log_dict["train/loss"] = loss.item()
            log_dict["train/perplexity"] = perplexity(loss).item()
            log_dict["train/lr"] = lr


        if model.config.use_moe:
            tokens_per_expert = aux["tokens_per_expert"]
            if model.config.use_moe and (step % train_config.log_moe_every == 0):
                layers_to_log = sorted(set([0, model.config.num_layers // 2, model.config.num_layers - 1]))
                for layer_idx in layers_to_log:
                    tpe = tokens_per_expert[layer_idx].detach().float().cpu().numpy()  # (E,)
                    msg = " | ".join([f"E{e}:{tpe[e]:.3f}" for e in range(len(tpe))])
                    if train_config.wandb_logging:
                        for e in range(len(tpe)):
                            log_dict[f"moe/layer_{layer_idx}_expert_{e}_tokens"] = tpe[e]

        if train_config.eval_log_interval > 0 and (step + 1) % train_config.eval_log_interval == 0:
            # Cleanup
            del inputs, targets, logits, loss
            clear_memory()

            eval_loss, eval_perplexity = eval_model(model, train_config)
            if train_config.wandb_logging:
                log_dict["eval/loss"] = eval_loss.item()
                log_dict["eval/perplexity"] = eval_perplexity.item()


            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                out_path = os.path.join(
                    train_config.save_checkpoint_dir,
                    train_config.model_name,
                    f"best_model_step_{step + 1}.pt",
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=step + 1,
                    out=out_path,
                    verbose=True,
                )

        # Sample generation
        if train_config.sampling_log_interval > 0 and (step + 1) % train_config.sampling_log_interval == 0:
            generated_outputs = generate(
                model=model,
                prompt="Once upon a time",
                tokenizer=tokenizer,
                max_new_tokens=256,
                top_k=50,
                temperature=0.8,
            )
            generated_text = generated_outputs["generated_text"]
            print("Once upon a time", end="")

        if train_config.wandb_logging and log_dict:
            wandb.log(log_dict, step=step + 1)


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    train_config: TrainingConfig,
):
    model.eval()

    eval_loss = 0.0
    eval_perplexity = 0.0
    # Load evaluation dataset
    original_data = np.memmap(
        train_config.eval_data_path,
        dtype=np.uint16,
        mode="r+",
    )
    x = torch.from_numpy(original_data)

    total_tokens = len(original_data)
    num_eval_batches = total_tokens // (train_config.batch_size * model.config.max_seq_len)

    state = BatchState(pos=0)
    with torch.no_grad():
        for _ in trange(num_eval_batches):
            inputs, targets = data_loading_sequential(
                x=x,
                batch_size=train_config.batch_size,
                context_length=model.config.max_seq_len,
                device=next(model.parameters()).device,
                state=state,
            )

            # Forward pass
            logits, aux = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = cross_entropy(logits, targets)

            eval_loss += loss.item()
            eval_perplexity += perplexity(loss).item()

    eval_loss = torch.tensor(eval_loss / num_eval_batches)
    eval_perplexity = torch.tensor(eval_perplexity / num_eval_batches)

    model.train()

    return eval_loss, eval_perplexity


def get_ctx(use_mixed: bool, device: torch.device, verbose: bool = True):
    if use_mixed and device.type == "cuda":

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        return nullcontext()