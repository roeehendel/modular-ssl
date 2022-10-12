import argparse
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from torchvision.transforms import transforms

import ssl_methods
from encoders.encoder import Encoder
from ssl_methods.components.heads.dino_head import DINOHead
from ssl_methods.components.losses.dino_loss import DINOLoss
from ssl_methods.ssl_method import SSLMethod
from transforms.branches_transform import BranchesTransform, TargetedMultiviewTransformPipeline
from transforms.multi_view.duplicate_transform import DuplicateTransform
from transforms.single_view.color_transform import ColorTransform
from transforms.single_view.spatial_transform import SpatialTransform
from transforms.single_view.standard_augmentations_transform import StandardAugmentationsTransform
from utils.nn_utils.module_deepcopy import module_deepcopy
from utils.nn_utils.freeze_module import freeze_module
from utils.nn_utils.momentum_update import momentum_update


@ssl_methods.registry.register("dino")
class DINO(SSLMethod):
    def __init__(self, encoder: Encoder, **kwargs):
        super().__init__(encoder, **kwargs)

        hparams = self.hparams

        head = DINOHead(input_dim=encoder.embedding_dim(), **hparams)

        self._student = nn.Sequential(OrderedDict(encoder=encoder, head=head))
        self._teacher = module_deepcopy(self._student)
        freeze_module(self._teacher)

        self.criterion = DINOLoss(**hparams)

        self._teacher_momentum_schedule = None

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("DINO")

        # Augmentations
        parent_parser = StandardAugmentationsTransform.add_argparse_args(parent_parser)
        parser.add_argument("--n_local_crops", type=int, default=0)

        # Architecture and Method
        parent_parser = DINOHead.add_argparse_args(parent_parser)
        parent_parser = DINOLoss.add_argparse_args(parent_parser)
        parser.add_argument("--teacher_momentum", type=float, default=0.996)

        # Optimization
        parser.set_defaults(optimizer="adamw")
        parser.set_defaults(base_lr=5e-4, weight_decay=0.04, warmup_epochs=10)

        return parent_parser

    def branches_views_transform(self, normalization: Optional = None) -> BranchesTransform:
        hparams = self.hparams

        transform_params = vars(hparams)
        transform_params.update({"normalization": normalization, "crop_size": hparams.input_height})

        global_views_transform = TargetedMultiviewTransformPipeline([
            DuplicateTransform(2),
            StandardAugmentationsTransform(**transform_params),
        ], target_branches=[0, 1])

        local_crop_factor = 96 / 224
        local_views_transform = TargetedMultiviewTransformPipeline([
            DuplicateTransform(hparams.n_local_crops),
            SpatialTransform(crop_size=int(hparams.input_height * local_crop_factor), crop_scale=(0.05, 0.3)),
            ColorTransform(),
            transforms.ToTensor(),
            normalization,
        ], target_branches=[0])

        transform_pipelines = [global_views_transform]
        if self.hparams.n_local_crops > 0:
            transform_pipelines.append(local_views_transform)

        return BranchesTransform(
            transform_pipelines=[global_views_transform, local_views_transform],
            num_branches=2
        )

    def forward(self, *args, **kwargs):
        return self._teacher.encoder(*args, **kwargs)

    def forward_branch(self, view: Tensor, branch_idx: int) -> Tensor:
        if branch_idx == 0:
            return self._student(view)
        elif branch_idx == 1:
            with torch.no_grad():
                return self._teacher(view)

    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        return self.criterion(branches_outputs, self.current_epoch)

    # TODO: make parameter scheduling and momentum update more general
    # TODO: (it should replace: on_fit_start and _teacher_momentum)
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
        momentum_update(source=self._student, target=self._teacher, momentum=self._teacher_momentum())


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
