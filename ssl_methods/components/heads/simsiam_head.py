from typing import Type

from torch import nn, Tensor

from utils.nn_utils.mlp import MLP


class SimSiamHead(nn.Module):
    def __init__(self, input_dim: int, projector_dim: int, predictor_hidden_dim: int, activation_fn: Type[nn.Module]):
        super().__init__()
        self.projector = MLP(input_dim=input_dim, hidden_dim=input_dim, output_dim=projector_dim,
                             num_layers=3, activation_fn=activation_fn)
        self.predictor = MLP(input_dim=projector_dim, hidden_dim=predictor_hidden_dim, output_dim=projector_dim,
                             activation_fn=activation_fn)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.projector(x)
        h = self.predictor(z)
        return z, h
