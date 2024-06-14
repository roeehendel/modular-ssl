import argparse
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
from pl_bolts.optimizers import linear_warmup_decay


class SSLMethod(pl.LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    def batch_size(self) -> int:
        return self.hparams.batch_size * self.trainer.num_devices

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
        warmup_steps = (hparams.warmup_epochs /
                        self.trainer.max_epochs) * total_steps

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    # TODO: replace linear_warmup_decay with a version that does not require pl_bolts
                    linear_warmup_decay(
                        warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        }

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)

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

    def __str__(self) -> str:
        return self.__class__.__name__
