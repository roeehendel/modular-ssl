import torch
from einops import repeat
from timm.models.vision_transformer import PatchEmbed, Block
from torch import nn

from utils.positional_embedding import get_2d_sincos_pos_embed


class MaskedVisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()

        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer) for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            add_cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_tokens(self, x, idx_keep):
        B, L, D = x.shape  # batch, length, dim
        idx_keep_repeated = repeat(idx_keep, 'b l -> b l d', d=D)  # [B, len_keep, D]
        x_masked = torch.gather(x, dim=1, index=idx_keep_repeated)
        return x_masked

    def add_positional_embeddings(self, x):
        return x + self.pos_embed[:, 1:, :]

    def add_cls_token(self, x):
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def apply_transformer(self, x):
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        idx_keep = None
        if len(x) == 2:
            x, idx_keep = x
        x = self.patch_embed(x)
        x = self.add_positional_embeddings(x)
        if idx_keep is not None:
            x = self.mask_tokens(x, idx_keep)
        x = self.add_cls_token(x)
        x = self.apply_transformer(x)

        return x


def masked_vit_tiny(img_size=224, patch_size=16, **kwargs):
    model = MaskedVisionTransformerEncoder(
        img_size=img_size, patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., **kwargs
    )
    return model


def masked_vit_small(img_size=224, patch_size=16, **kwargs):
    model = MaskedVisionTransformerEncoder(
        img_size=img_size, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs
    )
    return model


def masked_vit_base(img_size=224, patch_size=16, **kwargs):
    model = MaskedVisionTransformerEncoder(
        img_size=img_size, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs
    )
    return model


def masked_vit_large(img_size=224, patch_size=16, **kwargs):
    model = MaskedVisionTransformerEncoder(
        img_size=img_size, patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs
    )
    return model
