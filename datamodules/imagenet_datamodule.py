import os
from argparse import ArgumentParser
from typing import Optional

import datamodules
from config import DATASETS_DIR
from datamodules.base_imagenet_datamodule import BaseImagenetDataModule
from utils.datasets.cached_image_folder import CachedImageFolder
from utils.datasets.random_subset_dataset import RandomSubsetDataset


@datamodules.registry.register('imagenet')
class ImagenetDataModule(BaseImagenetDataModule):
    def __init__(self, data_dir: str, seed: int = 42, eval_train_subset_size: Optional[float] = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.seed = seed
        self.eval_train_subset_size = eval_train_subset_size

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_path = os.path.join(self.data_dir, 'train')
            val_path = os.path.join(self.data_dir, 'val')

            image_folder_class = CachedImageFolder

            self.pretrain_dataset = image_folder_class(root=train_path, transform=self.pretrain_transform)
            self.eval_train_dataset = image_folder_class(root=train_path, transform=self.eval_train_transform)
            if self.eval_train_subset_size < 1.0:
                self.eval_train_dataset = RandomSubsetDataset(self.eval_train_dataset, self.eval_train_subset_size)
            self.eval_val_dataset = image_folder_class(root=val_path, transform=self.eval_val_transform)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)

        parser.add_argument("--dataset_dir", type=str, default=os.path.join(DATASETS_DIR, 'ilsvrc'))

        return parent_parser
