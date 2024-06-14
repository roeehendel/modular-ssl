import argparse
from argparse import ArgumentParser
from typing import Type

from pl_bolts.models.self_supervised.resnets import BasicBlock, Bottleneck
from pl_bolts.models.self_supervised.resnets import ResNet as _ResNet
from torch import nn

import encoders
from encoders.encoder import Encoder

_VARIANTS_KWARGS = {
    "18": dict(block=BasicBlock, layers=[2, 2, 2, 2]),
    "34": dict(block=BasicBlock, layers=[3, 4, 6, 3]),
    "50": dict(block=Bottleneck, layers=[3, 4, 6, 3]),
    "101": dict(block=Bottleneck, layers=[3, 4, 23, 3]),
}


@encoders.registry.register("resnet")
class ResNet(_ResNet, Encoder):
    def __init__(self, variant: str, cifar_stem: bool = False, *args, **kwargs):
        super().__init__(first_conv=not cifar_stem, maxpool1=not cifar_stem, **_VARIANTS_KWARGS[variant])
        self.variant = variant

    def forward(self, x):
        return super().forward(x)[0]

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)

        print('hola')

        parser.add_argument("--variant", type=str, default="18", choices=_VARIANTS_KWARGS.keys())
        parser.add_argument("--cifar_stem", action=argparse.BooleanOptionalAction)

        temp_args, _ = parent_parser.parse_known_args()
        cifar_stem = temp_args.dataset in ["cifar10"]
        parser.set_defaults(cifar_stem=cifar_stem)

        return parent_parser

    def embedding_dim(self) -> int:
        return self.fc.in_features

    def activation_fn(self) -> Type[nn.Module]:
        return nn.ReLU

    def full_name(self) -> str:
        return f"resnet_{self.variant}"
