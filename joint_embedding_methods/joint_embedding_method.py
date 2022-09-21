from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


def knn_predict(embeddings, train_embeddings, train_labels, k: int = 1):
    batch_size = embeddings.size(0)

    dist = torch.mm(embeddings, train_embeddings.T)
    yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
    candidates = train_labels.view(1, -1).expand(batch_size, -1)
    predictions = torch.gather(candidates, 1, yi)
    predictions = predictions.narrow(1, 0, 1).clone().view(-1)

    return predictions


class JointEmbeddingMethod(pl.LightningModule, ABC):
    def __init__(self, encoder: nn.Module, knn_k: int = 20, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder

        self.accuracy = torchmetrics.Accuracy()

        self._train_embeddings = []
        self._train_labels = []

    @abstractmethod
    def branch1(self, view):
        pass

    @abstractmethod
    def branch2(self, view):
        pass

    @abstractmethod
    def head(self, out1, out2):
        pass

    @property
    def batch_size(self):
        return self.trainer.datamodule.train_dataloader().batch_size * self.trainer.num_devices

    def forward(self, views):
        multiview, eval_view = views
        view1, view2 = multiview
        out1, out2 = self.branch1(view1), self.branch2(view2)
        loss = self.head(out1, out2)

        with torch.no_grad():
            self.encoder.eval()
            embeddings = self.encoder(eval_view)
            self.encoder.train()
        return loss, embeddings

    def on_train_epoch_start(self) -> None:
        self._train_embeddings = []
        self._train_labels = []

    def training_step(self, batch, batch_idx):
        views, labels = batch
        loss, embeddings = self(views)

        self.log_dict({"train_loss": loss})

        # TODO: replace the list with a pre-initialized tensor
        embeddings = F.normalize(embeddings, dim=1)
        self._train_embeddings.append(embeddings.detach().cpu())
        self._train_labels.append(labels.detach().cpu())

        return loss

    def on_validation_epoch_start(self) -> None:
        if len(self._train_embeddings):
            self._train_embeddings = torch.cat(self._train_embeddings, dim=0).to(self.device)
            self._train_labels = torch.cat(self._train_labels, dim=0).to(self.device)

    def validation_step(self, batch, batch_idx):
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
            self.log('val_acc', self.accuracy, on_step=True, on_epoch=True, prog_bar=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("JointEmbeddingPretraining")
        return parent_parser
