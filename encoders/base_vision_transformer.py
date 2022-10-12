from abc import ABC
from argparse import ArgumentParser

import datamodules
from encoders.encoder import Encoder

VARIANTS_KWARGS = {
    'debug': dict(embed_dim=16, num_heads=2, depth=2, mlp_ratio=1),
    'cifar10': dict(embed_dim=512, num_heads=8, depth=6, mlp_ratio=1),
    'tiny': dict(embed_dim=192, num_heads=3, depth=12, mlp_ratio=4),
    'small': dict(embed_dim=384, num_heads=6, depth=12, mlp_ratio=4),
    'base': dict(embed_dim=768, num_heads=12, depth=12, mlp_ratio=4),
    'large': dict(embed_dim=1024, num_heads=16, depth=24, mlp_ratio=4),
}


class BaseViT(Encoder, ABC):
    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group("ViT")

        parser.add_argument("--variant", type=str, default="small", choices=VARIANTS_KWARGS.keys())
        parser.add_argument("--img_size", type=int)
        parser.add_argument("--patch_size", type=int)

        temp_args, _ = parent_parser.parse_known_args()

        dataset_class = datamodules.registry.get(temp_args.dataset)
        img_size = dataset_class.img_dims.height

        patch_sizes = {'cifar10': 4, 'stl10': 16, 'imagenet': 32}
        default_patch_size = int(img_size / 14) * 2  # This give 4 for CIFAR10 and 32 for ImageNet
        patch_size = patch_sizes.get(temp_args.dataset, default_patch_size)

        parser.set_defaults(img_size=img_size, patch_size=patch_size)

        return parent_parser
