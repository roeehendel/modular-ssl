from functools import partial

import torch
from einops import repeat
from timm.models.vision_transformer import VisionTransformer, checkpoint_seq
from torch import nn


class MaskedVisionTransformer(VisionTransformer):
    def mask_tokens(self, x, idx_keep):
        B, L, D = x.shape  # batch, length, dim
        idx_keep_repeated = repeat(idx_keep, 'b l -> b l d', d=D)  # [B, len_keep, D]
        x_masked = torch.gather(x, dim=1, index=idx_keep_repeated)
        return x_masked

    def forward_features(self, x):
        idx_keep = None
        if len(x) == 2:
            x, idx_keep = x

        x = self.patch_embed(x)
        x = self._pos_embed(x)

        if idx_keep is not None:
            x = self.mask_tokens(x, idx_keep)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x


class MaskedVisionTransformerEncoder(MaskedVisionTransformer):
    def forward(self, x):
        x = self.forward_features(x)
        return x


def vit_tiny_masked(img_size: int = 224, patch_size: int = 16, **kwargs):
    model = MaskedVisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=0,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
