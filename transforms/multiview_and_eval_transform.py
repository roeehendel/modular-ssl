from typing import Callable

from transforms.multi_view.multiview_transform import MultiviewTransform


class MultiviewAndEvalTransform(MultiviewTransform):
    def __init__(self, multiview_transform: Callable, eval_transform: Callable) -> None:
        self.multiview_transform = multiview_transform
        self.eval_transform = eval_transform

    def __call__(self, sample):
        return self.multiview_transform(sample), self.eval_transform(sample)
