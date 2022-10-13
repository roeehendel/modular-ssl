import argparse
from argparse import ArgumentParser

from timm.models import BasicBlock, Bottleneck
from timm.models import ResNet as TimmResNet
from torch import nn

import encoders
from encoders.encoder import Encoder

_VARIANTS_KWARGS = {
    "18": dict(block=BasicBlock, layers=[2, 2, 2, 2]),
    "34": dict(block=BasicBlock, layers=[3, 4, 6, 3]),
    "50": dict(block=Bottleneck, layers=[3, 4, 6, 3]),
    "101": dict(block=Bottleneck, layers=[3, 4, 23, 3]),
}


@encoders.registry.register("timm_resnet")
class TimmResNet(TimmResNet, Encoder):
    def __init__(self, variant: str, cifar_stem: bool = False, **kwargs):
        super().__init__(num_classes=0, **_VARIANTS_KWARGS[variant])
        self.variant = variant
        if cifar_stem:
            in_channels, out_channels = self.conv1.in_channels, self.conv1.out_channels
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)

        parser.add_argument("--variant", type=str, default="18", choices=_VARIANTS_KWARGS.keys())
        parser.add_argument("--cifar_stem", action=argparse.BooleanOptionalAction)

        temp_args, _ = parent_parser.parse_known_args()
        cifar_stem = temp_args.dataset in ["cifar10"]
        parser.set_defaults(cifar_stem=cifar_stem)

        return parent_parser

    def embedding_dim(self) -> int:
        return self.num_features

    def __str__(self) -> str:
        return f"timm_resnet_{self.variant}"
