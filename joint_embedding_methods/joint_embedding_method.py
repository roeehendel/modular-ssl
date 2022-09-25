import argparse
from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
from pl_bolts.optimizers import linear_warmup_decay
from torch import nn, Tensor

from transforms.multi_view.branches_views_transform import BranchesViewsTransform


class JointEmbeddingMethod(pl.LightningModule, ABC):
    def __init__(self, encoder: nn.Module, embedding_dim: int, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder

    @abstractmethod
    def branches_views_transform(self, input_height: int, normalization: Optional = None) -> BranchesViewsTransform:
        pass

    @abstractmethod
    def forward_branch(self, view: Tensor, branch_idx: int) -> Tensor:
        pass

    @abstractmethod
    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        pass

    @property
    def batch_size(self) -> int:
        return self.trainer.datamodule.train_dataloader().batch_size * self.trainer.num_devices

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def forward_branches(self, branches_views: list[list[Tensor]]) -> list[list[Tensor]]:
        branches_outputs = []
        for branch_idx, branch_views in enumerate(branches_views):
            branch_outputs = []
            for view in branch_views:
                branch_outputs.append(self.forward_branch(view, branch_idx))
            branches_outputs.append(branch_outputs)

        return branches_outputs

    def training_step(self, batch, batch_idx) -> Tensor:
        views, labels = batch
        branches_views, eval_view = views

        branches_outputs = self.forward_branches(branches_views)
        loss = self.forward_loss(branches_outputs)
        self.log_dict({"train_loss": loss})

        return loss

    def validation_step(self, *args, **kwargs) -> None:
        pass

    def configure_optimizers(self):
        hparams = self.hparams

        lr = hparams.base_lr * self.batch_size / 256.0

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
        else:
            raise ValueError(f"Unknown optimizer: {hparams.optimizer}")

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
        parser = parent_parser.add_argument_group('JointEmbeddingMethod')

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

        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--warmup_epochs', type=int, default=0)

        return parent_parser
