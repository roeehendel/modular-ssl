from abc import ABC, abstractmethod
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

    def full_name(self) -> str:
        return self.__class__.__name__
