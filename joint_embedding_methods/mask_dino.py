from typing import Optional

from joint_embedding_methods.dino import DINO
from transforms.multi_view.branches_views_transform import BranchesViewsTransform
from transforms.multi_view.patch_masking_multiview_transform import PatchMaskingMultiviewTransform
from transforms.single_view.simclr_transform import SimCLRTransform


class MaskDINO(DINO):
    def branches_views_transform(self, input_height: int, normalization: Optional = None) -> BranchesViewsTransform:
        hparams = self.hparams

        view_transform = SimCLRTransform(
            input_height=input_height,
            gaussian_blur=hparams.gaussian_blur,
            jitter_strength=hparams.jitter_strength,
            normalize=normalization,
            crop_scale=(1.0, 1.0)
        )

        multiview_transform = PatchMaskingMultiviewTransform(self.encoder.patch_embed.num_patches, mask_ratio=0.75)

        return BranchesViewsTransform(
            shared_preprocess_transform=view_transform,
            shared_views_transforms=[multiview_transform],
            branches_views_transforms=[[], []],
            shared_postprocess_transform=view_transform
        )
