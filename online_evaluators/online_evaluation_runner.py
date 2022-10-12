from typing import Sequence, Optional, Any

import torch
from pytorch_lightning import LightningModule, Trainer, Callback

from online_evaluators.online_evalutaor import OnlineEvaluator
from utils.set_training import set_training


class OnlineEvaluationRunner(Callback):
    """ Performs online evaluation of a representation learning algorithm. """

    def __init__(self, embed_dim: int, online_evaluators: list[OnlineEvaluator]):
        super().__init__()

        self.embed_dim = embed_dim
        self.online_evaluators = online_evaluators

        self.train_embeddings = None
        self.train_labels = None

        self.val_embeddings = None
        self.val_labels = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        for online_evaluator in self.online_evaluators:
            for named_metric in online_evaluator.metrics:
                setattr(pl_module, named_metric.name, named_metric.metric)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for online_evaluator in self.online_evaluators:
            online_evaluator.device = pl_module.device

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_embeddings = []
        self.train_labels = []
        self.val_embeddings = []
        self.val_labels = []

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs: Sequence, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.current_epoch == 0:
            return

        if dataloader_idx == 0:
            self._create_and_save_embeddings(pl_module, batch, stage='train')
        else:
            self._create_and_save_embeddings(pl_module, batch, stage='val')

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch == 0:
            return

        self.train_embeddings = torch.cat(self.train_embeddings, dim=0).to(pl_module.device)
        self.train_labels = torch.cat(self.train_labels, dim=0).to(pl_module.device)

        for online_evaluator in self.online_evaluators:
            online_evaluator.on_train_epoch_end(self.train_embeddings, self.train_labels)

        self.val_embeddings = torch.cat(self.val_embeddings, dim=0)
        self.val_labels = torch.cat(self.val_labels, dim=0)

        for online_evaluator in self.online_evaluators:
            online_evaluator.on_validation_epoch_end(self.val_embeddings, self.val_labels)
            for named_metric in online_evaluator.metrics:
                pl_module.log(named_metric.name, named_metric.metric, on_step=False, on_epoch=True, prog_bar=True)

    def _create_and_save_embeddings(self, pl_module: LightningModule, batch: Sequence, stage: str) -> None:
        images, labels = batch
        embeddings = self._get_embeddings(pl_module, images)

        embeddings = embeddings.detach().cpu()
        labels = labels.detach().cpu()

        if stage == 'train':
            self.train_embeddings.append(embeddings)
            self.train_labels.append(labels)
        elif stage == 'val':
            self.val_embeddings.append(embeddings)
            self.val_labels.append(labels)

    @staticmethod
    def _get_embeddings(pl_module: LightningModule, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), set_training(pl_module, False):
            embeddings = pl_module(images)
        return embeddings
