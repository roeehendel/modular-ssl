from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Type

from torch import nn


class Encoder(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def activation_fn(self) -> Type[nn.Module]:
        pass

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        # parser = parent_parser.add_argument_group(cls.__name__)

        return parent_parser

    def __str__(self) -> str:
        return self.__class__.__name__
