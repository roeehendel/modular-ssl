import torch
from torch import Tensor


def batch_randperm(batch_size: int, num_patches: int, device: torch.device) -> Tensor:
    return torch.argsort(torch.rand(batch_size, num_patches, device=device), dim=-1)
