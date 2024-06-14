import argparse

import torch
from torch import nn, Tensor, distributed as dist
from torch.nn import functional as F


class DINOLoss(nn.Module):
    def __init__(self, output_dim: int,
                 teacher_start_temp: float, teacher_temp: float, teacher_temp_warmup_epochs: int, student_temp: float,
                 center_momentum: float, **kwargs):
        super().__init__()
        self.student_temp = student_temp

        self.teacher_start_temp = teacher_start_temp
        self.teacher_temp = teacher_temp
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs

        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, output_dim))

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)

        parser.add_argument("--teacher_start_temp", type=float, default=0.04)
        parser.add_argument("--teacher_temp", type=float, default=0.07)
        parser.add_argument("--teacher_temp_warmup_epochs", type=int, default=30)
        parser.add_argument("--student_temp", type=float, default=0.1)
        parser.add_argument("--center_momentum", type=float, default=0.9)

        return parent_parser

    def forward(self, branches_outputs: list[list[Tensor]], epoch: int):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_output, teacher_output = branches_outputs

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
