from torch import nn


class LambdaModule(nn.Module):
    def __init__(self, lamb):
        super().__init__()
        import types
        assert type(lamb) is types.LambdaType
        self._lambda = lamb

    def forward(self, x):
        return self._lambda(x)
