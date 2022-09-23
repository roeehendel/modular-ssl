from functools import partial

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


def vit_super_tiny(img_size=32, patch_size=4, **kwargs):
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=0, embed_dim=384, depth=7, num_heads=12, mlp_ratio=2,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
