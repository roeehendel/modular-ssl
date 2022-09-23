from abc import ABC, abstractmethod


class MultiviewTransform(ABC):
    @abstractmethod
    def __call__(self, sample) -> list:
        pass
