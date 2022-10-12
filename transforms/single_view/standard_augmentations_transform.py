import argparse
from typing import Optional

from torchvision.transforms import transforms

from transforms.single_view.color_transform import ColorTransform
from transforms.single_view.spatial_transform import SpatialTransform


class StandardAugmentationsTransform(object):
    def __init__(self, normalization: Optional = None, **kwargs) -> None:
        spatial_transform = SpatialTransform(**kwargs)
        color_transform = ColorTransform(**kwargs)

        transform_list = [spatial_transform, color_transform, transforms.ToTensor()]
        if normalization is not None:
            transform_list.append(normalization)

        self.transform = transforms.Compose(transform_list)

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = SpatialTransform.add_argparse_args(parent_parser)
        parent_parser = ColorTransform.add_argparse_args(parent_parser)
        return parent_parser

    def __call__(self, sample):
        return self.transform(sample)
