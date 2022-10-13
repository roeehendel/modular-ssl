import argparse
from abc import ABC, abstractmethod
from typing import Callable, Optional, Type
from torch import Tensor

import encoders
from encoders.encoder import Encoder
from transforms.branches_transform import BranchesTransform

from ssl_methods.ssl_method import SSLMethod


class MultiviewSSLMethod(SSLMethod, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = self._create_encoder(kwargs)

    @abstractmethod
    def pretrain_transform(self, normalization: Optional[Callable] = None) -> BranchesTransform:
        pass

    @abstractmethod
    def forward_branch(self, view: Tensor, branch_idx: int) -> Tensor:
        pass

    @abstractmethod
    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        pass

    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim()

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def forward_branches(self, branches_views: list[list[Tensor]]) -> list[list[Tensor]]:
        branches_outputs = []
        for branch_idx, branch_views in enumerate(branches_views):
            branch_outputs = []
            for _, view in enumerate(branch_views):
                branch_outputs.append(self.forward_branch(view, branch_idx))
            branches_outputs.append(branch_outputs)

            # Code to forward all views together (doesn't work with current SimSiam impl. which returns a tuple)
            # batch_size = branch_views[0].shape[0]
            # combined_views = torch.cat(branch_views, dim=0)
            # combined_outputs = self.forward_branch(combined_views, branch_idx)
            # branches_outputs.append(torch.split(combined_outputs, batch_size, dim=0))

        return branches_outputs

    def training_step(self, batch, batch_idx, **kwargs) -> Tensor:
        branches_views, labels = batch
        branches_outputs = self.forward_branches(branches_views)
        loss = self.forward_loss(branches_outputs)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def _create_encoder(args) -> Encoder:
        encoder_constructor: Type[Encoder] = encoders.registry.get(args['encoder_type'])
        encoder = encoder_constructor(**args)
        return encoder

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)

        encoders_list = encoders.registry.keys()
        parser.add_argument("--encoder_type", type=str, default="resnet", help="Encoder type", choices=encoders_list)

        temp_args, _ = parent_parser.parse_known_args()
        encoder_class: Type[Encoder] = encoders.registry.get(temp_args.encoder_type)
        parent_parser = encoder_class.add_argparse_args(parent_parser)

        # parser.add_argument('--combine_views_in_forward', action=argparse.BooleanOptionalAction)
        # parser.set_defaults(combine_views_in_forward=False)

        return parent_parser

    def __str__(self) -> str:
        return f"{self.__class__.__name__}-{self.encoder}"
