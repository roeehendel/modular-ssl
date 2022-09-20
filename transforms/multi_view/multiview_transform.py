from abc import ABC, abstractmethod


class MultiviewTransform(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple:
        pass
