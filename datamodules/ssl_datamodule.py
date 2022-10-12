from abc import ABC
from argparse import ArgumentParser
from typing import Callable, Optional, NamedTuple, Union

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision.transforms import transforms

from transforms.single_view.evaluation_transform import EvaluationTransform


class ImageDims(NamedTuple):
    channels: int
    height: int
    width: int


class SSLDataModule(LightningDataModule, ABC):
    img_dims: ImageDims
    num_classes: int
    normalization_mean: tuple
    normalization_std: tuple

    def __init__(self, batch_size: int, val_batch_size: Optional[int] = None, num_workers: int = 0, **kwargs):
        if val_batch_size is None:
            self.val_batch_size = batch_size * 4

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pretrain_dataset = None
        self.eval_train_dataset = None
        self.eval_val_dataset = None

        self._pretrain_dataloader = None
        self._eval_train_dataloader = None
        self._eval_val_dataloader = None

        self._pretrain_transform = None
        self._eval_train_transform = None
        self._eval_val_transform = None

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        # parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__class__.__name__)

        parser.add_argument("--batch_size", type=int, default=128, help="batch size per gpu")
        parser.add_argument("--val_batch_size", type=int, default=None, help="batch size per gpu for validation")
        parser.add_argument("--num_workers", type=int, default=8, help="num of workers per GPU")
        parser.add_argument("--input_height", type=int,
                            default=cls.img_dims.height if cls.img_dims is not None else None)
        parser.add_argument("--dataset_num_classes", type=int,
                            default=cls.num_classes if cls.num_classes is not None else None)

        return parent_parser

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.pretrain_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [self.eval_train_dataloader, self.eval_val_dataloader]

    @property
    def pretrain_dataloader(self) -> Union[DataLoader, IterableDataset]:
        if self._pretrain_dataloader is None:
            self._pretrain_dataloader = self._dataloader(self.pretrain_dataset, shuffle=True,
                                                         batch_size=self.batch_size)
        return self._pretrain_dataloader

    @property
    def eval_train_dataloader(self) -> Union[DataLoader, IterableDataset]:
        if self._eval_train_dataloader is None:
            self._eval_train_dataloader = self._dataloader(self.eval_train_dataset, batch_size=self.val_batch_size)
        return self._eval_train_dataloader

    @property
    def eval_val_dataloader(self) -> Union[DataLoader, IterableDataset]:
        if self._eval_val_dataloader is None:
            self._eval_val_dataloader = self._dataloader(self.eval_val_dataset, batch_size=self.val_batch_size)
        return self._eval_val_dataloader

    @property
    def pretrain_transform(self) -> Callable:
        if self._pretrain_transform is None:
            return self.default_transforms
        return self._pretrain_transform

    @pretrain_transform.setter
    def pretrain_transform(self, transform: Callable):
        self._pretrain_transform = transform

    @property
    def eval_train_transform(self) -> Callable:
        if self._eval_train_transform is None:
            return self.default_transforms
        return self._eval_train_transform

    @eval_train_transform.setter
    def eval_train_transform(self, transform: Callable):
        self._eval_train_transform = transform

    @property
    def eval_val_transform(self) -> Callable:
        if self._eval_val_transform is None:
            return self.default_transforms
        return self._eval_val_transform

    @eval_val_transform.setter
    def eval_val_transform(self, transform: Callable):
        self._eval_val_transform = transform

    @property
    def default_transforms(self) -> Callable:
        return EvaluationTransform(input_height=self.img_dims.height, normalization=self.normalization_transform)

    @property
    def normalization_transform(self) -> Optional[Callable]:
        return transforms.Normalize(self.normalization_mean, self.normalization_std)

    def _dataloader(self, dataset: Dataset,
                    batch_size: int, shuffle: bool = False) -> Union[DataLoader, IterableDataset]:
        shuffle &= not isinstance(dataset, IterableDataset)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size, num_workers=self.num_workers,
            shuffle=shuffle, pin_memory=True, persistent_workers=self.num_workers > 0
        )
