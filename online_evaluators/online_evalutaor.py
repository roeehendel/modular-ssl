from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor
from torchmetrics import Metric


@dataclass
class NamedMetric:
    name: str
    metric: Metric


class OnlineEvaluator(ABC):
    def __init__(self):
        self._device = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    @abstractmethod
    def metrics(self) -> list[NamedMetric]:
        pass

    @abstractmethod
    def on_train_epoch_end(self, train_embeddings: Tensor, train_labels: Tensor):
        pass

    @abstractmethod
    def on_validation_epoch_end(self, validation_embeddings: Tensor, validation_labels: Tensor):
        pass
