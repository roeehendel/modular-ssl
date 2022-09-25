import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets


class ImagenetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, drop_last: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            print(self.train_transforms)
            self.train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'),
                                                      transform=self.train_transforms)
            self.val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'),
                                                    transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
