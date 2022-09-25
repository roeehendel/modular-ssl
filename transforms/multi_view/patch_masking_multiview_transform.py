from typing import List

import torch

from transforms.multi_view.multiview_transform import MultiviewTransform


class PatchMaskingMultiviewTransform(MultiviewTransform):
    def __init__(self, num_patches: int, mask_ratio: float = 0.75) -> None:
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio

    def __call__(self, sample) -> List:
        num_patches = self.num_patches
        num_keep = int(num_patches * (1 - self.mask_ratio))

        idx_keep = torch.randperm(num_patches, device=sample.device)[:num_keep * 2]
        idx_keep1 = idx_keep[:num_keep]
        idx_keep2 = idx_keep[num_keep:]
        idx_keep1 = idx_keep1.sort()[0]
        idx_keep2 = idx_keep2.sort()[0]

        return [(sample, idx_keep1), (sample, idx_keep2)]
