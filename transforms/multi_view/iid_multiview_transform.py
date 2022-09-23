from typing import Callable

from torch import Tensor

from transforms.multi_view.multiview_transform import MultiviewTransform


class IIDMultiviewTransform(MultiviewTransform):
    def __init__(self, transform: Callable, n_transforms: int = 1) -> None:
        self.transform = transform
        self.n_transforms = n_transforms

    def __call__(self, sample) -> list[Tensor]:
        return [self.transform(sample) for _ in range(self.n_transforms)]
