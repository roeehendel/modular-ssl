import argparse
from typing import Optional

import torch
from pl_bolts.optimizers import linear_warmup_decay
from torch import nn, Tensor

from joint_embedding_methods.joint_embedding_method import JointEmbeddingMethod
from transforms.multi_view.branches_views_transform import BranchesViewsTransform
from transforms.multi_view.iid_multiview_transform import IIDMultiviewTransform
from transforms.single_view.simclr_transform import SimCLRTransform


class SimSiam(JointEmbeddingMethod):
    def __init__(self, encoder: nn.Module, **kwargs):
        super().__init__(encoder, **kwargs)

        hparams = self.hparams

        self._branch = SimSiamBranch(encoder, hparams.embedding_dim, hparams.hidden_size, hparams.output_dim)
        self._loss = SimSiamLoss()

    def branches_views_transform(self, input_height: int, normalization: Optional = None) -> BranchesViewsTransform:
        hparams = self.hparams

        view_transform = SimCLRTransform(
            input_height=input_height,
            gaussian_blur=hparams.gaussian_blur,
            jitter_strength=hparams.jitter_strength,
            normalize=normalization,
        )

        multiview_transform = IIDMultiviewTransform(view_transform, n_transforms=2)

        return BranchesViewsTransform(
            shared_views_transforms=[multiview_transform],
        )

    def forward_branch(self, view: Tensor, branch_idx: int) -> Tensor:
        return self._branch(view)

    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        return self._loss(branches_outputs)

    def configure_optimizers(self):
        hparams = self.hparams

        lr = hparams.base_lr * self.batch_size / 256

        if hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay
            )
        elif hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr, weight_decay=hparams.weight_decay
            )
        elif hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr, weight_decay=hparams.weight_decay
            )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = (hparams.warmup_epochs / self.trainer.max_epochs) * total_steps

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

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group('SimSiam')

        # Augmentations
        parser.add_argument("--gaussian_blur", action=argparse.BooleanOptionalAction)
        parser.add_argument("--jitter_strength", type=float, default=0.5)
        parser.set_defaults(gaussian_blur=False)

        # Architecture
        parser.add_argument('--hidden_size', type=int, default=2048)
        parser.add_argument('--output_dim', type=int, default=2048)

        # Optimization
        parser.add_argument('--optimizer', type=str, default='sgd')
        temp_args, _ = parent_parser.parse_known_args()
        if temp_args.optimizer == 'sgd':
            parser.add_argument('--base_lr', type=float, default=0.025)
            parser.add_argument('--momentum', type=float, default=0.9)
        elif temp_args.optimizer == 'adamw':
            parser.add_argument('--base_lr', type=float, default=5e-5)
        elif temp_args.optimizer == 'adam':
            parser.add_argument('--base_lr', type=float, default=5e-5)
        else:
            raise ValueError(f"Unknown optimizer {temp_args.optimizer}")

        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--warmup_epochs', type=int, default=0)

        return parent_parser


class SimSiamBranch(nn.Module):
    def __init__(self, encoder, encoder_output_dim, hidden_size, output_dim):
        super().__init__()
        self.projector = MLP(encoder_output_dim, hidden_size, output_dim)
        self.predictor = MLP(output_dim, hidden_size // 4, output_dim)
        self.encoder = encoder

    def forward(self, view):
        y = self.encoder(view)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, branches_outputs):
        branch_outputs = branches_outputs[0]

        _, z1, h1 = branch_outputs[0]
        _, z2, h2 = branch_outputs[1]

        z1, z2 = z1.detach(), z2.detach()

        loss = -(self.criterion(h1, z2).mean() + self.criterion(h2, z1).mean()) * 0.5

        return loss


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int) -> None:
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
