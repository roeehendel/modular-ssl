import argparse
from collections import OrderedDict
from typing import Callable, Optional

from torch import nn, Tensor

import ssl_methods
from encoders.encoder import Encoder
from ssl_methods.components.heads.vicreg_head import VICRegHead
from ssl_methods.components.losses.vicreg_loss import VICRegLoss
from ssl_methods.multiview_ssl_method import MultiviewSSLMethod
from transforms.branches_transform import BranchesTransform, TargetedMultiviewTransformPipeline
from transforms.multi_view.duplicate_transform import DuplicateTransform
from transforms.single_view.standard_augmentations_transform import StandardAugmentationsTransform


@ssl_methods.registry.register("vicreg")
class VICReg(MultiviewSSLMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        head = VICRegHead(
            input_dim=self.encoder.embedding_dim(),
            projector_dim=self.hparams.projector_dim,
            activation_fn=self.encoder.activation_fn(),
        )

        self._branch = nn.Sequential(OrderedDict(encoder=self.encoder, head=head))

        self._loss = VICRegLoss()

    def pretrain_transform(self, normalization: Optional[Callable] = None) -> BranchesTransform:
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

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)

        # Augmentations
        parent_parser = StandardAugmentationsTransform.add_argparse_args(parent_parser)
        parser.add_argument('--num_views', type=int, default=2)

        # Architecture
        parser.add_argument('--projector_dim', type=int, default=8192)

        # Optimization
        parent_parser.set_defaults(optimizer='sgd')
        parser.set_defaults(base_lr=5e-5, momentum=0.9, weight_decay=1e-6, warmup_epochs=10)

        return parent_parser
