from typing import List

from transforms.multi_view.multiview_transform_pipeline import MultiviewTransformPipeline


class TargetedMultiviewTransformPipeline(MultiviewTransformPipeline):
    def __init__(self, transforms: list, target_branches: List[int]) -> None:
        super().__init__(transforms)
        self.target_branches = target_branches


class BranchesTransform(object):
    def __init__(self, transform_pipelines: List[TargetedMultiviewTransformPipeline], num_branches: int = 2) -> None:
        self.transform_pipelines = transform_pipelines
        self.num_branches = num_branches

    def __call__(self, sample):
        branch_samples = [[] for _ in range(self.num_branches)]
        for transform_pipeline in self.transform_pipelines:
            samples = transform_pipeline(sample)
            for branch in transform_pipeline.target_branches:
                branch_samples[branch] += samples
        return branch_samples
