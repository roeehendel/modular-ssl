from functools import partial
from typing import Type

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

import encoders
from encoders.base_vision_transformer import VARIANTS_KWARGS, BaseViT


@encoders.registry.register("vit")
class ViT(VisionTransformer, BaseViT):
    def __init__(self, variant: str, img_size: int, patch_size: int, **kwargs):
        self.variant = variant
        super().__init__(
            img_size=img_size, patch_size=patch_size, num_classes=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **VARIANTS_KWARGS[variant],
        )
        self.num_patches = self.patch_embed.num_patches

    def embedding_dim(self) -> int:
        return self.embed_dim

    def activation_fn(self) -> Type[nn.Module]:
        return nn.GELU

    def full_name(self) -> str:
        return f"vit_{self.variant}"
