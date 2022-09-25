import argparse
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import trunc_normal_

from joint_embedding_methods.joint_embedding_method import JointEmbeddingMethod
from joint_embedding_methods.utils.freeze_module import freeze_module
from joint_embedding_methods.utils.momentum_update import momentum_update
from transforms.multi_view.branches_views_transform import BranchesViewsTransform
from transforms.multi_view.iid_multiview_transform import IIDMultiviewTransform
from transforms.single_view.simclr_transform import SimCLRTransform
from utils.nn_module_deepcopy import copy_module_with_weight_norm


class DINO(JointEmbeddingMethod):
    def __init__(self, encoder: nn.Module, **kwargs):
        super().__init__(encoder, **kwargs)

        hparams = self.hparams

        self.student = nn.Sequential(
            encoder,
            DINOHead(hparams.embedding_dim, hparams.out_dim, hparams.use_bn_in_head, hparams.norm_last_layer)
        )
        self.teacher = copy_module_with_weight_norm(self.student)
        freeze_module(self.teacher)

        self.criterion = DINOLoss(
            out_dim=hparams.out_dim,
            ncrops=2 + hparams.n_crops,  # total number of crops = 2 global crops + local_crops_number
            teacher_start_temp=hparams.teacher_start_temp,
            teacher_temp=hparams.teacher_temp,
            teacher_temp_warmup_epochs=hparams.teacher_temp_warmup_epochs
        )

        # niters_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        # self.momentum_schedule = cosine_scheduler(teacher_momentum, 1, self.trainer.max_epochs, niters_per_epoch,
        #                                           len(self.trainer.train_dataloader))

    def branches_views_transform(self, input_height: int, normalization: Optional = None) -> BranchesViewsTransform:
        hparams = self.hparams

        view_transform = SimCLRTransform(
            input_height=input_height,
            gaussian_blur=hparams.gaussian_blur,
            jitter_strength=hparams.jitter_strength,
            normalize=normalization,
            crop_scale=(0.2, 1.0)
        )

        multicrop_transform = SimCLRTransform(
            input_height=input_height,
            gaussian_blur=hparams.gaussian_blur,
            jitter_strength=hparams.jitter_strength,
            normalize=normalization,
            crop_scale=(0.1, 0.5)
        )

        multiview_transform = IIDMultiviewTransform(view_transform, n_transforms=2)
        multicrop_transforms = IIDMultiviewTransform(multicrop_transform, n_transforms=hparams.n_crops)

        return BranchesViewsTransform(
            shared_views_transforms=[multiview_transform],
            branches_views_transforms=[[], [multicrop_transforms]],
        )

    def training_step_end(self, step_output):
        momentum_update(self.student, self.teacher, self._teacher_momentum())

    def _teacher_momentum(self):
        # TODO: implement cosine scheduler that goes from 0.996 to 1
        return self.hparams.teacher_momentum

    def forward_branch(self, view: Tensor, branch_idx: int) -> Tensor:
        if branch_idx == 0:
            return self.student(view)
        elif branch_idx == 1:
            with torch.no_grad():
                return self.teacher(view)

    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        student_output, teacher_output = branches_outputs
        loss = self.criterion(student_output, teacher_output, self.current_epoch)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("DINO")

        # Augmentations
        parser.add_argument("--gaussian_blur", action=argparse.BooleanOptionalAction)
        parser.add_argument("--jitter_strength", type=float, default=0.5)
        parser.add_argument("--n_crops", type=int, default=0)
        parser.set_defaults(gaussian_blur=False)

        # Architecture
        parser.add_argument("--out_dim", type=int, default=4096)
        parser.add_argument("--use_bn_in_head", action=argparse.BooleanOptionalAction)
        parser.add_argument("--norm_last_layer", action=argparse.BooleanOptionalAction)
        parser.set_defaults(use_bn_in_head=False, norm_last_layer=True)

        # Method
        parser.add_argument("--teacher_start_temp", type=float, default=0.04)
        parser.add_argument("--teacher_temp", type=float, default=0.07)
        parser.add_argument("--teacher_temp_warmup_epochs", type=int, default=30)
        parser.add_argument("--teacher_momentum", type=float, default=0.996)

        # Optimization
        parser.set_defaults(optimizer="adamw")
        parser.set_defaults(base_lr=5e-4, weight_decay=0.04, warmup_epochs=10)

        return parent_parser


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, teacher_start_temp, teacher_temp,
                 teacher_temp_warmup_epochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp

        self.teacher_start_temp = teacher_start_temp
        self.teacher_temp = teacher_temp
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs

        self.center_momentum = center_momentum

        self.ncrops = ncrops

        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: list[Tensor], teacher_output: list[Tensor], epoch: int):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_output = torch.stack(student_output)
        student_output = student_output / self.student_temp
        student_output = list(student_output)

        # teacher centering and sharpening
        temp = self._teacher_temp(epoch)
        teacher_output_centered_sharpened = list(F.softmax((torch.stack(teacher_output) - self.center) / temp, dim=-1))

        total_loss = 0
        n_loss_terms = 0
        for it, q in enumerate(teacher_output_centered_sharpened):
            for iv, v in enumerate(student_output):
                if iv == it:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def _teacher_temp(self, current_epoch):
        # we apply a warmup for the teacher temperature because
        # a too high temperature makes the training unstable at the beginning
        if current_epoch < self.teacher_temp_warmup_epochs:
            progress = current_epoch / self.teacher_temp_warmup_epochs
            return self.teacher_start_temp + (self.teacher_temp - self.teacher_start_temp) * progress
        else:
            return self.teacher_temp

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        teacher_output = torch.cat(teacher_output)
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        world_size = 1
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            world_size = dist.get_world_size()
        batch_center = batch_center / (len(teacher_output) * world_size)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


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
