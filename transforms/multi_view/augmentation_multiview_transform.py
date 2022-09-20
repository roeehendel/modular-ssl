from typing import Callable

from transforms.multi_view.multiview_transform import MultiviewTransform


class AugmentationMultiviewTransform(MultiviewTransform):
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, sample):
        return self.transform(sample), self.transform(sample)
