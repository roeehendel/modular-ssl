import os.path
from argparse import ArgumentParser
from typing import Union

import torch
import torch.distributed as dist
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, default_collate

import datamodules
from datamodules.base_imagenet_datamodule import BaseImagenetDataModule


def identity(x):
    return x


@datamodules.registry.register('wds_imagenet')
class WDSImagenetDataModule(BaseImagenetDataModule):
    def __init__(self, batch_size: int, shards=None, valshards=None, bucket=None, **kwargs):
        super().__init__(batch_size, **kwargs)

        self.training_urls = os.path.join(bucket, shards)
        self.val_urls = os.path.join(bucket, valshards)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)

        parser.add_argument("--bucket", default="/home/gamir/hendel/datasets/ilsvrc/shards")
        parser.add_argument("--shards", default="imagenet-train-{000000..000146}.tar")
        parser.add_argument("--valshards", default="imagenet-val-{000000..000006}.tar")

        return parent_parser

    def _make_dataloader(self, urls, transform, mode="train"):
        if isinstance(urls, str) and urls.startswith("fake:"):
            xs = torch.randn((self.batch_size, 3, 224, 224))
            ys = torch.zeros(self.batch_size, dtype=torch.int64)
            return wds.MockDataset((xs, ys), 10000)

        if mode == "train":
            dataset_size = 1_281_167
            shuffle = 5000
        elif mode == "val":
            dataset_size = 5000
            shuffle = 0

        dataset = (
            wds.WebDataset(urls, nodesplitter=wds.split_by_worker)
            .shuffle(shuffle)
            .decode("pil")
            .to_tuple("jpg;png;jpeg cls")
            .map_tuple(transform, identity)
            .batched(self.batch_size, collation_fn=default_collate, partial=False)
        )

        def collate(x):
            if isinstance(x[0], list):
                x[0] = default_collate(x[0])
            return x[0], torch.from_numpy(x[1])

        # loader = DataLoader(
        #     dataset,
        #     batch_size=None,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     collate_fn=collate,
        # )
        # loader.length = dataset_size // self.batch_size

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=collate,
        )
        loader.length = dataset_size // self.batch_size
        loader.batch_size = self.batch_size

        if mode == "train" and dist.is_initialized():
            # ensure same number of batches in all clients
            total_num_workers = self.num_workers * dist.get_world_size()
            dataset_total_batches = loader.length
            batches_per_worker = dataset_total_batches // total_num_workers
            batches_per_worker = 1  # TODO: remove!!!
            equalized_total_batches = batches_per_worker * total_num_workers
            loader = loader.with_length(equalized_total_batches)  # .repeat(2)

        return loader

    @property
    def pretrain_dataloader(self) -> Union[DataLoader, IterableDataset]:
        return self._make_dataloader(self.training_urls, self.pretrain_transform, mode="train")

    @property
    def eval_train_dataloader(self) -> Union[DataLoader, IterableDataset]:
        return self._make_dataloader(self.training_urls, self.eval_train_transform, mode="train")

    @property
    def eval_val_dataloader(self) -> Union[DataLoader, IterableDataset]:
        return self._make_dataloader(self.val_urls, self.eval_val_transform, mode="val")
