import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pl_bolts.optimizers import linear_warmup_decay
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.utils.weight_norm import WeightNorm

from joint_embedding_methods.joint_embedding_method import JointEmbeddingMethod


class DINO(JointEmbeddingMethod):
    def __init__(self, encoder: nn.Module, encoder_output_dim: int,
                 out_dim: int = 65536, use_bn_in_head: bool = False, norm_last_layer: bool = True,
                 teacher_start_temp: float = 0.04, teacher_temp: float = 0.07, teacher_temp_warmup_epochs: int = 30,
                 teacher_momentum: float = 0.996,
                 base_lr: float = 5e-2, weight_decay: float = 0.00, warmup_epochs: int = 0,
                 # base_lr: float = 5e-4, weight_decay: float = 0.04, warmup_epochs: int = 10,
                 **kwargs):
        super().__init__(encoder, **kwargs)

        self.student = nn.Sequential(
            encoder,
            DINOHead(encoder_output_dim, out_dim, use_bn_in_head, norm_last_layer)
        )
        self.teacher = copy_module_with_weight_norm(self.student)
        # for p in self.teacher.parameters():
        #     p.requires_grad = False

        self.criterion = DINOLoss(
            out_dim=out_dim,
            ncrops=2,  # total number of crops = 2 global crops + local_crops_number
            teacher_start_temp=teacher_start_temp,
            teacher_temp=teacher_temp,
            teacher_temp_warmup_epochs=teacher_temp_warmup_epochs
        )

        # niters_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        # self.momentum_schedule = cosine_scheduler(teacher_momentum, 1, self.trainer.max_epochs, niters_per_epoch,
        #                                           len(self.trainer.train_dataloader))

    def training_step_end(self, step_output):
        with torch.no_grad():
            m = self._teacher_momentum()
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def _teacher_momentum(self):
        # TODO: implement cosine scheduler that goes from 0.996 to 1
        return self.hparams.teacher_momentum

    def branch1(self, view):
        return self.student(view)

    def branch2(self, view):
        with torch.no_grad():
            return self.teacher(view)

    def head(self, out1, out2):
        loss = self.criterion(out1, out2, self.current_epoch)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.base_lr * self.batch_size / 256

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = (self.hparams.warmup_epochs / self.trainer.max_epochs) * total_steps

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        }


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

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self._teacher_temp(epoch)
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        print(teacher_out.shape)
        print(student_out.shape)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
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
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def copy_module_with_weight_norm(original_module: nn.Module):
    for module in original_module.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                delattr(module, hook.name)
    copy_module = copy.deepcopy(original_module)
    for module in original_module.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook(module, None)
    return copy_module


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
