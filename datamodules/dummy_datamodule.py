from argparse import ArgumentParser
from typing import Optional

import torch
from torch.utils.data import TensorDataset

import datamodules
from datamodules.ssl_datamodule import SSLDataModule, ImageDims


@datamodules.registry.register('dummy')
class DummyDataModule(SSLDataModule):
    img_dims = None
    num_classes = None
    normalization_mean = (0.5, 0.5, 0.5)
    normalization_std = (0.25, 0.25, 0.25)

    def __init__(self, train_samples: int, img_dims: ImageDims, **kwargs):
        super().__init__(**kwargs)

        self.train_samples = train_samples
        self.__class__.img_dims = img_dims

        self._sample = torch.randn(1, *self.img_dims)
        self._train_dataset = TensorDataset(self._sample.expand(self.train_samples, *self.img_dims))

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__class__.__name__)

        parser.add_argument("--train_samples", type=int, default=1_281_167, help="number of train samples to use")
        parser.add_argument("--img_dims", type=ImageDims, default=ImageDims(3, 224, 224), help="image dimensions")

        return parent_parser

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.pretrain_dataset = self._train_dataset
            self.eval_train_dataset = self._train_dataset
            self.eval_val_dataset = self._train_dataset
