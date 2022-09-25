from functools import partial

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


def vit_micro(img_size: int = 224, patch_size: int = 16, **kwargs):
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=0,
        embed_dim=384, depth=7, num_heads=12, mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_tiny(img_size: int = 224, patch_size: int = 16, **kwargs):
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=0,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
