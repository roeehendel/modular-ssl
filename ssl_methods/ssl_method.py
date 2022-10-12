import argparse
from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
from pl_bolts.optimizers import linear_warmup_decay
from torch import Tensor

from encoders.encoder import Encoder
from transforms.branches_transform import BranchesTransform


class SSLMethod(pl.LightningModule, ABC):
    def __init__(self, encoder: Encoder, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder

    @abstractmethod
    def branches_views_transform(self, normalization: Optional = None) -> BranchesTransform:
        pass

    @abstractmethod
    def forward_branch(self, view: Tensor, branch_idx: int) -> Tensor:
        pass

    @abstractmethod
    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        pass

    def batch_size(self) -> int:
        return self.hparams.batch_size * self.trainer.num_devices

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def forward_branches(self, branches_views: list[list[Tensor]]) -> list[list[Tensor]]:
        branches_outputs = []
        for branch_idx, branch_views in enumerate(branches_views):
            branch_outputs = []
            for view_idx, view in enumerate(branch_views):
                branch_outputs.append(self.forward_branch(view, branch_idx))
            branches_outputs.append(branch_outputs)

            # Code to forward all views together (doesn't work with current SimSiam impl. which returns a tuple)
            # batch_size = branch_views[0].shape[0]
            # combined_views = torch.cat(branch_views, dim=0)
            # combined_outputs = self.forward_branch(combined_views, branch_idx)
            # branches_outputs.append(torch.split(combined_outputs, batch_size, dim=0))

        return branches_outputs

    def training_step(self, batch, batch_idx) -> Tensor:
        branches_views, labels = batch
        branches_outputs = self.forward_branches(branches_views)
        loss = self.forward_loss(branches_outputs)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, *args, **kwargs) -> None:
        pass

    def configure_optimizers(self):
        hparams = self.hparams

        lr = hparams.base_lr * self.batch_size() / 256.0

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
                    # TODO: replace linear_warmup_decay with a version that does not require pl_bolts
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        }

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group('SSLMethod')

        # parser.add_argument('--combine_views_in_forward', action=argparse.BooleanOptionalAction)
        # parser.set_defaults(combine_views_in_forward=False)

        parser.add_argument('--optimizer', type=str, default='sgd')
        parser.add_argument('--base_lr', type=float)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--warmup_epochs', type=int, default=0)

        temp_args, _ = parent_parser.parse_known_args()
        if temp_args.optimizer == 'sgd':
            parser.set_defaults(base_lr=0.03)
            parser.add_argument('--momentum', type=float, default=0.9)
        elif temp_args.optimizer == 'adamw':
            parser.set_defaults(base_lr=5e-4, weight_decay=0.01)
        elif temp_args.optimizer == 'adam':
            parser.set_defaults(base_lr=5e-4)
        else:
            raise ValueError(f"Unknown optimizer {temp_args.optimizer}")

        return parent_parser
