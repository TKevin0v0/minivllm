from __future__ import annotations
import concurrent.futures, glob, os
from contextlib import contextmanager
import torch
from safetensors.torch import load_file
from torch import nn
from tqdm import tqdm
__all__ = ["default_dtype_context", "default_weight_loader", "load_model", "safetensors_weights_iterator", "skip_param_init"]
def default_weight_loader(param, loaded_weight): param.data.copy_(loaded_weight)
@contextmanager
def skip_param_init():
    linear, embed = nn.Linear.reset_parameters, nn.Embedding.reset_parameters
    nn.Linear.reset_parameters = lambda self: None; nn.Embedding.reset_parameters = lambda self: None
    try: yield
    finally: nn.Linear.reset_parameters, nn.Embedding.reset_parameters = linear, embed
@contextmanager
def default_dtype_context(dtype):
    prev = torch.get_default_dtype(); torch.set_default_dtype(dtype)
    try: yield
    finally: torch.set_default_dtype(prev)
def safetensors_weights_iterator(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files: raise FileNotFoundError(f"No safetensors files found in {folder}")
    for f in tqdm(files, desc="Loading weights"):
        yield from load_file(f, device="cpu").items()
def load_model(model, folder): model.load_weights(safetensors_weights_iterator(folder)); return model
