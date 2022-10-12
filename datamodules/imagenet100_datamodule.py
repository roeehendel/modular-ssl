import os
from argparse import ArgumentParser
from typing import Optional

import datamodules
from config import DATASETS_DIR
from datamodules.imagenet_datamodule import ImagenetDataModule


@datamodules.registry.register('imagenet100')
class Imagenet100DataModule(ImagenetDataModule):
    def __init__(self, data_dir: str, seed: int = 42, eval_train_subset_size: Optional[float] = 0.2, **kwargs):
        super().__init__(data_dir, seed, eval_train_subset_size, **kwargs)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = super().add_argparse_args(parent_parser, **kwargs)
        parser.set_defaults(data_dir=os.path.join(DATASETS_DIR, 'ilsvrc100'))
        return parser
