from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn, Tensor

from transforms.multi_view.branches_views_transform import BranchesViewsTransform


def knn_predict(embeddings, train_embeddings, train_labels, k: int = 1):
    batch_size = embeddings.size(0)

    dist = torch.mm(embeddings, train_embeddings.T)
    yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
    candidates = train_labels.view(1, -1).expand(batch_size, -1)
    predictions = torch.gather(candidates, 1, yi)
    predictions = predictions.narrow(1, 0, 1).clone().view(-1)

    return predictions


class JointEmbeddingMethod(pl.LightningModule, ABC):
    def __init__(self, encoder: nn.Module, embedding_dim: int, knn_k: int = 20, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder

        self.accuracy = torchmetrics.Accuracy()

        self._train_embeddings = []
        self._train_labels = []

    @abstractmethod
    def branches_views_transform(self, input_height: int, normalization: Optional = None) -> BranchesViewsTransform:
        pass

    @abstractmethod
    def forward_branch(self, view: Tensor, branch_idx: int) -> Tensor:
        pass

    @abstractmethod
    def forward_loss(self, branches_outputs: list[list[Tensor]]) -> Tensor:
        pass

    @property
    def batch_size(self) -> int:
        return self.trainer.datamodule.train_dataloader().batch_size * self.trainer.num_devices

    def forward_branches(self, branches_views: list[list[Tensor]]) -> list[list[Tensor]]:
        branches_outputs = []
        for branch_idx, branch_views in enumerate(branches_views):
            branch_outputs = []
            for view in branch_views:
                branch_outputs.append(self.forward_branch(view, branch_idx))
            branches_outputs.append(branch_outputs)

        return branches_outputs

    def forward_eval_embeddings(self, eval_view) -> Tensor:
        with torch.no_grad():
            self.encoder.eval()
            embeddings = self.encoder(eval_view)
            self.encoder.train()
        return embeddings

    def on_train_epoch_start(self) -> None:
        self._train_embeddings = []
        self._train_labels = []

    def training_step(self, batch, batch_idx) -> Tensor:
        views, labels = batch
        branches_views, eval_view = views

        branches_outputs = self.forward_branches(branches_views)
        loss = self.forward_loss(branches_outputs)
        self.log_dict({"train_loss": loss})

        # TODO: replace the list with a pre-initialized tensor
        embeddings = self.forward_eval_embeddings(eval_view)
        embeddings = F.normalize(embeddings, dim=1)
        self._train_embeddings.append(embeddings.detach().cpu())
        self._train_labels.append(labels.detach().cpu())

        return loss

    def on_validation_epoch_start(self) -> None:
        if len(self._train_embeddings):
            self._train_embeddings = torch.cat(self._train_embeddings, dim=0).to(self.device)
            self._train_labels = torch.cat(self._train_labels, dim=0).to(self.device)

    def validation_step(self, batch, batch_idx) -> None:
        if len(self._train_embeddings):
            views, labels = batch

            embeddings = self.encoder(views)
            labels = labels

            # predictions = knn_predict(embeddings, self._train_embeddings, self._train_labels, k=self.hparams.knn_k)

            batch_size = embeddings.size(0)
            dist = torch.mm(embeddings, self._train_embeddings.T)
            yd, yi = dist.topk(self.hparams.knn_k, dim=1, largest=True, sorted=True)
            candidates = self._train_labels.view(1, -1).expand(batch_size, -1)
            predictions = torch.gather(candidates, 1, yi)
            predictions = predictions.narrow(1, 0, 1).clone().view(-1)

            self.accuracy(predictions, labels)
            self.log('val_acc', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
