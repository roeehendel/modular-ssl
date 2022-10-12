import argparse
from collections import OrderedDict
from typing import Optional

from torch import nn, Tensor

import ssl_methods
from encoders.encoder import Encoder
from ssl_methods.components.heads.simsiam_head import SimSiamHead
from ssl_methods.components.losses.simsiam_loss import SimSiamLoss
from ssl_methods.ssl_method import SSLMethod
from transforms.branches_transform import BranchesTransform, TargetedMultiviewTransformPipeline
from transforms.multi_view.duplicate_transform import DuplicateTransform
from transforms.single_view.standard_augmentations_transform import StandardAugmentationsTransform


@ssl_methods.registry.register("simsiam")
class SimSiam(SSLMethod):
    def __init__(self, encoder: Encoder, **kwargs):
        super().__init__(encoder, **kwargs)

        head = SimSiamHead(
            input_dim=encoder.embedding_dim(),
            projector_dim=self.hparams.projector_dim,
            predictor_hidden_dim=self.hparams.predictor_hidden_dim,
            activation_fn=encoder.activation_fn(),
        )

        self._branch = nn.Sequential(OrderedDict(encoder=encoder, head=head))

        self._loss = SimSiamLoss()

    def branches_views_transform(self, normalization: Optional = None) -> BranchesTransform:
        hparams = self.hparams

        transform_params = vars(hparams)
        transform_params.update({"normalization": normalization, "crop_size": hparams.input_height})

        views_transform = TargetedMultiviewTransformPipeline([
            DuplicateTransform(self.hparams.num_views),
            StandardAugmentationsTransform(**transform_params),
        ], target_branches=[0])

        return BranchesTransform(
            transform_pipelines=[views_transform],
            num_branches=1
        )

    def forward_branch(self, views: Tensor, branch_idx: int) -> Tensor:
        return self._branch(views)

    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        return self._loss(branches_outputs)

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group('SimSiam')

        # Augmentations
        parent_parser = StandardAugmentationsTransform.add_argparse_args(parent_parser)
        parser.add_argument('--num_views', type=int, default=2)

        # Architecture
        parser.add_argument('--projector_dim', type=int, default=2048)
        parser.add_argument('--predictor_hidden_dim', type=int, default=512)

        # Optimization
        parent_parser.set_defaults(optimizer='sgd')
        parser.set_defaults(base_lr=0.025, momentum=0.9, weight_decay=5e-4, warmup_epochs=0)

        return parent_parser
