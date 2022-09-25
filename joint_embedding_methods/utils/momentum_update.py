import torch
from torch import nn


def momentum_update(source: nn.Module, target: nn.Module, momentum: float):
    with torch.no_grad():
        for param_q, param_k in zip(source.parameters(), target.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
