import torch
from pl_bolts.optimizers import linear_warmup_decay
from torch import nn, Tensor

from joint_embedding_methods.joint_embedding_method import JointEmbeddingMethod


class SimSiam(JointEmbeddingMethod):
    def __init__(self,
                 encoder: nn.Module, encoder_output_dim: int,
                 hidden_size: int = 2048, output_dim: int = 2048,
                 base_lr: float = 0.025, momentum: float = 0.9, weight_decay: float = 5e-4, warmup_epochs: int = 0,
                 **kwargs):
        super().__init__(encoder, **kwargs)

        self.projector = MLP(encoder_output_dim, hidden_size, output_dim)
        self.predictor = MLP(output_dim, hidden_size // 4, output_dim)
        self.criterion = nn.CosineSimilarity(dim=1)

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

    def _branch(self, view):
        y = self.encoder(view)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h

    def configure_optimizers(self):
        lr = self.hparams.base_lr * self.batch_size / 256

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = (self.hparams.warmup_epochs / self.trainer.max_epochs) * total_steps

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        }


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
