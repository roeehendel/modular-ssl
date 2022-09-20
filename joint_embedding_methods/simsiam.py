from torch import nn, Tensor

from joint_embedding_methods.joint_embedding_method import JointEmbeddingMethod


class SimSiam(JointEmbeddingMethod):
    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int = 2048,
                 hidden_size: int = 2048,
                 output_dim: int = 2048,
                 **kwargs):
        super().__init__(encoder, **kwargs)

        self.projector = MLP(input_dim, hidden_size, output_dim)
        self.predictor = MLP(output_dim, hidden_size // 4, output_dim)
        self.criterion = nn.CosineSimilarity(dim=1)

    def _branch(self, view):
        y = self.encoder(view)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h

    def branch1(self, view):
        return self._branch(view)

    def branch2(self, view):
        return self._branch(view)

    def head(self, out1, out2):
        _, z1, h1 = out1
        _, z2, h2 = out2

        z1, z2 = z1.detach(), z2.detach()

        loss = -(self.criterion(h1, z2).mean() + self.criterion(h2, z1).mean()) * 0.5

        return loss


class MLP(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_size: int = 2048, output_dim: int = 2048) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x
