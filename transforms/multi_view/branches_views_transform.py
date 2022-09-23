import itertools
from typing import Callable, Optional

from transforms.multi_view.multiview_transform import MultiviewTransform


class BranchesViewsTransform(Callable):
    def __init__(self, shared_preprocess_transform: Optional[Callable] = None,
                 shared_views_transforms: Optional[list[MultiviewTransform]] = (),
                 branches_views_transforms: Optional[list[list[MultiviewTransform]]] = ((),),
                 branches_postprocess_transform: Optional[list[Callable]] = None,
                 shared_postprocess_transform: Optional[Callable] = None
                 ) -> None:
        self.shared_preprocess_transform = shared_preprocess_transform
        self.shared_views_transforms = shared_views_transforms
        self.branches_views_transforms = branches_views_transforms
        self.branches_postprocess_transforms = branches_postprocess_transform
        self.shared_postprocess_transform = shared_postprocess_transform

    @staticmethod
    def process_multiview_transforms_list(multiview_transforms: list[MultiviewTransform], sample) -> list:
        views_per_transform = [transform(sample) for transform in multiview_transforms]
        views = list(itertools.chain(*views_per_transform))
        return views

    def __call__(self, sample) -> list[list]:
        if self.shared_preprocess_transform is not None:
            sample = self.shared_preprocess_transform(sample)

        shared_views = self.process_multiview_transforms_list(self.shared_views_transforms, sample)
        unique_branches_views = [self.process_multiview_transforms_list(branch_transforms, sample)
                                 for branch_transforms in self.branches_views_transforms]

        branches_views = [unique_branch_views + shared_views for unique_branch_views in unique_branches_views]

        # perform postprocess transforms on each branch
        if self.branches_postprocess_transforms is not None:
            branches_views = [[branch_postprocess_transform(view) for view in branch_views]
                              for branch_views, branch_postprocess_transform
                              in zip(branches_views, self.branches_postprocess_transforms)]

        # perform shared postprocess transform on all views
        if self.shared_postprocess_transform is not None:
            branches_views = [[self.shared_postprocess_transform(view) for view in branch_views]
                              for branch_views in branches_views]

        return branches_views
