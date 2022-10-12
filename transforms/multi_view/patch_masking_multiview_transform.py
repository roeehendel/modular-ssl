from typing import List, Optional, Tuple

import torch
from torch import Tensor

from transforms.multi_view.multiview_transform_pipeline import MultiviewTransform


def _get_mask_indices(num_patches: int, patches_per_view: float, device: Optional = None) -> Tuple[Tensor, Tensor]:
    num_keep = int(num_patches * patches_per_view)

    if 2 * num_keep > num_patches:
        raise ValueError('Too many patches per view: {} * {} > {}'.format(2, num_keep, num_patches))

    idx_randperm = torch.randperm(num_patches, device=device)
    idx_keep1 = idx_randperm[:num_keep]
    idx_keep2 = idx_randperm[num_keep:2 * num_keep]
    idx_keep1, idx_keep2 = idx_keep1.sort()[0], idx_keep2.sort()[0]

    return idx_keep1, idx_keep2


class PatchMaskingMultiviewTransform(MultiviewTransform):
    def __init__(self, num_patches: int, patches_per_view: float = 0.25) -> None:
        self.num_patches = num_patches
        self.patches_per_view = patches_per_view

    def __call__(self, samples: list) -> List:
        idx_keep1, idx_keep2 = _get_mask_indices(self.num_patches, self.patches_per_view, device=samples[0].device)
        return [(samples[0], idx_keep1, idx_keep2), (samples[1], idx_keep2, idx_keep1)]
