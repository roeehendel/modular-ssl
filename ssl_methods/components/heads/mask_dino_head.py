import argparse

from torch import nn

from encoders.masked_vision_transformer import MaskedDecoderViT
from ssl_methods.components.heads.dino_head import DINOHead


class MaskDINOStudentHead(nn.Module):
    def __init__(self, dino_head: nn.Module, **kwargs):
        super().__init__()
        self.decoder = MaskedDecoderViT(**kwargs)
        self.predictor = dino_head

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = DINOHead.add_argparse_args(parent_parser)
        return parent_parser

    def forward(self, x):
        encoder_out = x
        decoder_pred = self.predictor(self.decoder(x))
        return encoder_out, decoder_pred
