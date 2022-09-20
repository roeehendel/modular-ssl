from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torchmetrics
from pl_bolts.optimizers import linear_warmup_decay
from torch import nn


class JointEmbeddingMethod(pl.LightningModule, ABC):
    def __init__(self, encoder: nn.Module,
                 base_lr: float = 0.06, momentum: float = 0.9, weight_decay: float = 5e-4, warmup_epochs: int = 0,
                 knn_k: int = 200):
        super().__init__()

        self.encoder = encoder

        self.base_lr = base_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.knn_k = knn_k
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

    def forward(self, views):
        multiview, eval_view = views
        view1, view2 = multiview
        out1, out2 = self.branch1(view1), self.branch2(view2)
        loss = self.head(out1, out2)

        with torch.no_grad():
            self.backbone.eval()
            embeddings = self.encoder(eval_view)
            self.backbone.train()
        return loss, embeddings

    def on_train_epoch_start(self) -> None:
        self._train_embeddings = []
        self._train_labels = []

    def training_step(self, batch, batch_idx):
        views, labels = batch
        loss, embeddings = self(views)

        self.log_dict({"train_loss": loss})

        # TODO: replace the list with a pre-initialized tensor
        self._train_embeddings.append(embeddings.detach().cpu())
        self._train_labels.append(labels.detach().cpu())

        return loss

    def on_validation_epoch_start(self) -> None:
        if len(self._train_embeddings):
            self._train_embeddings = torch.cat(self._train_embeddings, dim=0)
            self._train_labels = torch.cat(self._train_labels, dim=0)

    def validation_step(self, batch, batch_idx):
        if len(self._train_embeddings):
            views, labels = batch
            batch_size = views.size(0)

            with torch.no_grad():
                embeddings = self.encoder(views).cpu()
            labels = labels.cpu()

            dist = torch.mm(embeddings, self._train_embeddings.T)
            yd, yi = dist.topk(self.knn_k, dim=1, largest=True, sorted=True)
            candidates = self._train_labels.view(1, -1).expand(batch_size, -1)
            predictions = torch.gather(candidates, 1, yi)

            predictions = predictions.narrow(1, 0, 1).clone().view(-1)

            self.accuracy(predictions, labels)
            self.log('val_acc', self.accuracy, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # batch_size = self.trainer.datamodule.train_dataloader().batch_size * self.trainer.num_devices
        # lr = self.base_lr * batch_size / 256

        lr = self.base_lr

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = (self.warmup_epochs / self.trainer.max_epochs) * total_steps

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("JointEmbeddingPretraining")
        return parent_parser
