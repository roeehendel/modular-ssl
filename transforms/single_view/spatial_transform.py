import argparse
from typing import Tuple

from torchvision.transforms import transforms


class SpatialTransform(object):
    def __init__(self, crop_size: int = 224, crop_scale: Tuple[float, float] = (0.2, 1.0), **kwargs) -> None:
        random_crop = transforms.RandomResizedCrop(size=crop_size, scale=crop_scale)
        random_flip = transforms.RandomHorizontalFlip(p=0.5)

        self.transform = transforms.Compose([
            random_crop,
            random_flip
        ])

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("SpatialTransform")

        parser.add_argument("--crop_size", type=int, default=224)
        parser.add_argument("--crop_scale", type=float, nargs='+', default=(0.2, 1.0))

        return parent_parser

    def __call__(self, sample):
        return self.transform(sample)
