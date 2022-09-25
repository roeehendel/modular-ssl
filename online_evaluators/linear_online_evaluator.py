import time
from typing import Optional

import torch
import torchmetrics
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from online_evaluators.online_evalutaor import OnlineEvaluator, NamedMetric


class LinearProbeOnlineEvaluator(OnlineEvaluator):
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol. """

    def __init__(self, embedding_dim: int, num_classes: int, drop_p: float = 0.0, hidden_dim: Optional[int] = None,
                 probe_training_epochs: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes: Optional[int] = num_classes
        self.drop_p = drop_p
        self.hidden_dim = hidden_dim
        self.probe_training_epochs = probe_training_epochs

        self.online_evaluator = SSLEvaluator(
            n_input=self.embedding_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        )
        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    @property
    def metrics(self) -> list[NamedMetric]:
        return [
            NamedMetric("probe_train_acc", self.train_accuracy),
            NamedMetric("probe_val_acc", self.val_accuracy)
        ]

    def on_train_epoch_end(self, train_embeddings: Tensor, train_labels: Tensor):
        batch_size = 512

        self.online_evaluator = self.online_evaluator.to(self.device)

        print("Training probe...")
        start_time = time.time()

        for i in range(self.probe_training_epochs):
            for batch_idx in range(0, len(train_embeddings), batch_size):
                batch_embeddings = train_embeddings[batch_idx:batch_idx + batch_size]
                batch_labels = train_labels[batch_idx:batch_idx + batch_size]
                self.optimizer.zero_grad()
                predictions, mlp_loss = self._shared_step(batch_embeddings, batch_labels)
                mlp_loss.backward()
                self.optimizer.step()

        print(f"Probe training took {time.time() - start_time} seconds")

        predictions, mlp_loss = self._shared_step(train_embeddings, train_labels)
        self.train_accuracy(predictions, train_labels)

    def on_validation_epoch_end(self, validation_embeddings: Tensor, validation_labels: Tensor):
        self.online_evaluator = self.online_evaluator.to(self.device)

        predictions, mlp_loss = self._shared_step(validation_embeddings, validation_labels)
        self.val_accuracy(predictions, validation_labels.to(self.device))

    def _shared_step(self, embeddings: Tensor, labels: Tensor):
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)

        mlp_logits = self.online_evaluator(embeddings)
        mlp_loss = F.cross_entropy(mlp_logits, labels)

        predictions = torch.argmax(mlp_logits, dim=-1)

        return predictions, mlp_loss


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
