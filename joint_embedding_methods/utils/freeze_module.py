from torch import nn


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False
