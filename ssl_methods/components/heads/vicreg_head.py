from typing import Type

from torch import nn, Tensor

from utils.nn_utils.mlp import MLP


class VICRegHead(nn.Module):
    def __init__(self, input_dim: int, projector_dim: int, activation_fn: Type[nn.Module]):
        super().__init__()
        self.projector = MLP(input_dim=input_dim, hidden_dim=projector_dim, output_dim=projector_dim,
                             num_layers=3, activation_fn=activation_fn)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.projector(x)
        return x
