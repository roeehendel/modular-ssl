from typing import Type

import torch
from einops import repeat
from timm.models.vision_transformer import checkpoint_seq
from torch import nn, Tensor

import encoders
from encoders.base_vision_transformer import BaseViT
from encoders.vision_transformer import ViT, VARIANTS_KWARGS


def _index_batch_of_sequences(sequences: Tensor, index: Tensor) -> Tensor:
    """
    Selects a subset of vectors from a batched tensor.
    """
    B, L, D = sequences.shape  # [batch, length, dim]
    index_expanded = repeat(index, 'b l -> b l d', d=D)  # [B, len_keep, D]
    indexed_sequences = torch.gather(sequences, dim=1, index=index_expanded)
    return indexed_sequences


@encoders.registry.register("masked_encoder_vit")
class MaskedEncoderViT(ViT):
    def __init__(self, variant: str, **kwargs):
        super().__init__(variant, **kwargs)

    def forward_features(self, x: Tensor, idx_keep: Tensor = None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        if idx_keep is not None:
            x = _index_batch_of_sequences(x, idx_keep)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def full_name(self) -> str:
        return f"masked_encoder_vit_{self.variant}"


class MaskedDecoderViT(BaseViT):
    def __init__(self, variant: str, img_size: int, patch_size: int, **kwargs):
        super().__init__()
        variant_kwargs = VARIANTS_KWARGS[variant]

        self._num_patches = (img_size // patch_size) ** 2

        self._embed_dim = variant_kwargs['embed_dim']
        self._num_heads = variant_kwargs['num_heads']
        self._mlp_hidden_dim = variant_kwargs['embed_dim'] * variant_kwargs['mlp_ratio']
        self._depth = variant_kwargs['depth']

        decoder_layer = nn.TransformerDecoderLayer(d_model=self._embed_dim, nhead=self._num_heads,
                                                   dim_feedforward=self._mlp_hidden_dim, activation=nn.GELU())
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        # self._query_token = nn.Parameter(torch.randn(self._embed_dim) * .02)
        self._pos_embed = nn.Parameter(torch.randn(self._num_patches, self._embed_dim) * .02)

    def forward(self, x: Tensor, input_idx: Tensor, output_idx: Tensor) -> Tensor:
        """
        @param x: [B, L, D] input tokens
        @param input_idx: [B, L] indices of input tokens
        @param output_idx: [B, L'] indices of output tokens
        """
        batch_size, num_outputs_tokens = output_idx.shape[:2]

        # base_queries = repeat(self._query_token, 'd -> b l d', b=batch_size, l=num_outputs_tokens)
        pos_embed_expanded = repeat(self._pos_embed, 'l d -> b l d', b=batch_size)
        pos_embed = _index_batch_of_sequences(pos_embed_expanded, output_idx)
        # queries = base_queries + pos_embed
        queries = pos_embed

        x = self._decoder(tgt=queries, memory=x)  # (tgt=queries, memory=x)

        return x

    def embedding_dim(self) -> int:
        return self._embed_dim

    def activation_fn(self) -> Type[nn.Module]:
        return nn.GELU
