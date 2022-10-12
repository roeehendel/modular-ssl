import argparse
from typing import Optional

import torch
from torch import Tensor
from torchvision.transforms import transforms

import ssl_methods
from encoders.encoder import Encoder
from encoders.masked_vision_transformer import MaskedDecoderViT
from ssl_methods.components.branches.mask_dino_student import MaskDINOStudent
from ssl_methods.components.branches.mask_dino_teacher import MaskDINOTeacher
from ssl_methods.components.heads.dino_head import DINOHead
from ssl_methods.components.losses.dino_loss import DINOLoss
from ssl_methods.dino import cosine_scheduler
from ssl_methods.ssl_method import SSLMethod
from transforms.branches_transform import BranchesTransform, TargetedMultiviewTransformPipeline
from transforms.multi_view.duplicate_transform import DuplicateTransform
from transforms.multi_view.patch_masking_multiview_transform import PatchMaskingMultiviewTransform
from transforms.single_view.color_transform import ColorTransform
from transforms.single_view.spatial_transform import SpatialTransform
from utils.nn_utils.module_deepcopy import module_deepcopy
from utils.nn_utils.freeze_module import freeze_module
from utils.nn_utils.momentum_update import momentum_update


@ssl_methods.registry.register("mask_dino")
class MaskDINO(SSLMethod):
    def __init__(self, encoder: Encoder, **kwargs):
        super().__init__(encoder, **kwargs)

        hparams = self.hparams

        decoder = MaskedDecoderViT(**hparams)
        head = DINOHead(input_dim=encoder.embedding_dim(), **hparams)

        self._student = MaskDINOStudent(encoder, decoder, head)
        self._teacher = MaskDINOTeacher(module_deepcopy(encoder), module_deepcopy(head))
        freeze_module(self._teacher)

        self.criterion = DINOLoss(**hparams)

        self.momentum_schedule = None

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("MaskDINO")

        # Augmentations
        parent_parser = SpatialTransform.add_argparse_args(parent_parser)
        parent_parser = ColorTransform.add_argparse_args(parent_parser)
        parser.add_argument("--patches_per_view", type=float, default=0.5)

        # Architecture and Method
        parent_parser.set_defaults(encoder_type='masked_encoder_vit')
        parent_parser = DINOHead.add_argparse_args(parent_parser)
        parent_parser = DINOLoss.add_argparse_args(parent_parser)
        parser.add_argument("--teacher_momentum", type=float, default=0.996)

        # Optimization
        parser.set_defaults(optimizer="adamw")
        parser.set_defaults(base_lr=5e-4, weight_decay=0.04, warmup_epochs=10)

        return parent_parser

    def branches_views_transform(self, normalization: Optional = None) -> BranchesTransform:
        hparams = self.hparams

        num_patches = self.encoder.patch_embed.num_patches

        transform_params = vars(hparams)
        transform_params.update({"crop_size": hparams.input_height, "crop_scale": (0.8, 1.0)})

        views_transform = TargetedMultiviewTransformPipeline([
            SpatialTransform(**transform_params),
            DuplicateTransform(2),
            ColorTransform(**transform_params),
            transforms.ToTensor(),
            normalization,
            PatchMaskingMultiviewTransform(num_patches, hparams.patches_per_view),
        ], target_branches=[0, 1])

        return BranchesTransform(
            transform_pipelines=[views_transform],
            num_branches=2
        )

    def forward_branch(self, view: Tensor, branch_idx: int) -> Tensor:
        images, input_idx, target_idx = view
        if branch_idx == 0:
            return self._student(images, input_idx, target_idx)
        elif branch_idx == 1:
            with torch.no_grad():
                return self._teacher(images, target_idx)

    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        # TODO: make the loss work per patch, this is temporary to test the rest of the code
        branches_outputs = [[out.mean(dim=1) for out in branch_outputs] for branch_outputs in branches_outputs]
        return self.criterion(branches_outputs, self.current_epoch)

    # TODO: make parameter scheduling and momentum update more general
    # TODO: (it should replace: on_fit_start, training_step_end and _teacher_momentum)
    def on_fit_start(self) -> None:
        niter_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        self._teacher_momentum_schedule = cosine_scheduler(
            base_value=self.hparams.teacher_momentum,
            final_value=1.0,
            epochs=self.trainer.max_epochs,
            niter_per_ep=niter_per_epoch
        )

    def _teacher_momentum(self):
        return self._teacher_momentum_schedule[self.global_step]

    def training_step_end(self, step_output):
        momentum_update(source=self._student.encoder, target=self._teacher.encoder, momentum=self._teacher_momentum())
        momentum_update(source=self._student.head, target=self._teacher.head, momentum=self._teacher_momentum())
