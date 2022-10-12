import argparse
from typing import Type

from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_

from utils.nn_utils.mlp import MLP


class DINOHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, bottleneck_dim: int, num_layers: int = 3,
                 use_bn: bool = False, norm_last_layer: bool = True, activation_fn: Type[nn.Module] = nn.GELU,
                 **kwargs):
        super().__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=bottleneck_dim,
                       num_layers=num_layers, activation_fn=activation_fn, use_bn=use_bn)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument("--output_dim", type=int, default=65536)
        parser.add_argument("--hidden_dim", type=int, default=2048)
        parser.add_argument("--bottleneck_dim", type=int, default=256)
        parser.add_argument("--use_bn", action=argparse.BooleanOptionalAction)
        parser.add_argument("--norm_last_layer", action=argparse.BooleanOptionalAction)
        parser.set_defaults(use_bn=False, norm_last_layer=True)
        return parent_parser

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
