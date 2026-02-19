import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    iteration,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    verbose: bool = False,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model, optimizer, verbose: bool = False
) -> int:
    state = torch.load(src, map_location=get_device())

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    return state["iteration"]


def get_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")