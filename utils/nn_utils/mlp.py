from functools import partial
from typing import Type

from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2,
                 activation_fn=nn.ReLU, use_bn: bool = True, drop_p: float = 0.):
        super().__init__()
        if num_layers == 1:
            self._model = nn.Linear(in_features=input_dim, out_features=output_dim)
        else:
            mlp_block_kwargs = dict(activation_fn=activation_fn, use_bn=use_bn, dropout=drop_p)
            mlp_block_constructor = partial(MLPBlock, **mlp_block_kwargs)
            layers = [mlp_block_constructor(input_dim, hidden_dim)] + \
                     [mlp_block_constructor(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] + \
                     [nn.Linear(hidden_dim, output_dim)]
            self._model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 activation_fn: Type[nn.Module] = partial(nn.ReLU, inplace=True),
                 use_bn: bool = False, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation_fn()
        self.use_bn = use_bn
        self.dropout = dropout
        if self.use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        if self.dropout > 0:
            x = self.drop(x)
        return x
