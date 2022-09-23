from typing import Callable

from torch import Tensor

from transforms.multi_view.branches_views_transform import BranchesViewsTransform


class TrainAndEvalTransform:
    def __init__(self, train_transform: BranchesViewsTransform, eval_transform: Callable) -> None:
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def __call__(self, sample) -> tuple[list[list], Tensor]:
        return self.train_transform(sample), self.eval_transform(sample)
