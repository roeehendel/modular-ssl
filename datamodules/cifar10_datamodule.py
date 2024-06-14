import os
from argparse import ArgumentParser
from typing import Optional, Union, Callable

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

import datamodules
from config import DATASETS_DIR
from datamodules.ssl_datamodule import SSLDataModule, ImageDims
from utils.datasets.splitting import split_dataset


@datamodules.registry.register('cifar10')
class CIFAR10DataModule(SSLDataModule):
    img_dims = ImageDims(3, 32, 32)
    num_classes = 10
    normalization_mean = (0.4914, 0.4822, 0.4465)
    normalization_std = (0.2471, 0.2435, 0.2616)

    def __init__(self, data_dir: str, val_split: Union[int, float], seed: int, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.val_split = val_split
        self.seed = seed

    @property
    def default_transforms(self) -> Callable:
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalization_transform
        ])

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)

        parser.set_defaults(
            data_dir=os.path.join(DATASETS_DIR, 'cifar10'),
            val_split=5000,
            seed=42,
        )

        return parent_parser

    def prepare_data(self, *args, **kwargs) -> None:
        CIFAR10(self.data_dir, train=True, download=True)
        # CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.pretrain_dataset = self._get_dataset(self.pretrain_transform)
            self.eval_train_dataset = self._get_dataset(self.eval_train_transform)
            self.eval_val_dataset = self._get_dataset(self.eval_val_transform, train=False)

    def _get_dataset(self, transform: Callable, train: bool = True) -> CIFAR10:
        idx = 0 if train else 1
        return split_dataset(
            CIFAR10(self.data_dir, train=True, transform=transform),
            self.val_split, self.seed
        )[idx]
