import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, imagenet_dir: str, transform):
        super().__init__()
        self.imagenet_dir = imagenet_dir
        self.transform = transform

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = datasets.ImageFolder(os.path.join(self.imagenet_dir, 'train'),
                                                      transform=self.transform)
            self.val_dataset = datasets.ImageFolder(os.path.join(self.imagenet_dir, 'val'),
                                                    transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=256,
            num_workers=10,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=512, num_workers=10)
