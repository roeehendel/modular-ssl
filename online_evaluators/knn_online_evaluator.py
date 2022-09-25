import torch
import torch.nn.functional as F
import torchmetrics

from online_evaluators.online_evalutaor import OnlineEvaluator, NamedMetric


class KNNOnlineEvaluator(OnlineEvaluator):
    """ Online evaluator using KNN"""

    def __init__(self, knn_k: int = 20, log_metric_name: str = 'knn_val_acc'):
        super().__init__()
        self.knn_k = knn_k
        self.log_metric_name = log_metric_name

        self.normalized_train_embeddings = None
        self.train_labels = None

        self.accuracy = torchmetrics.Accuracy()

    @property
    def metrics(self) -> list[NamedMetric]:
        return [NamedMetric(self.log_metric_name, self.accuracy)]

    def on_train_epoch_end(self, train_embeddings, train_labels):
        self.normalized_train_embeddings = F.normalize(train_embeddings, dim=1)
        self.train_labels = train_labels

    def on_validation_epoch_end(self, validation_embeddings, validation_labels):
        train_embeddings = self.normalized_train_embeddings.to(self.device)
        train_labels = self.train_labels.to(self.device)
        validation_embeddings = validation_embeddings.to(self.device)
        validation_labels = validation_labels.to(self.device)

        batch_size = validation_embeddings.size(0)
        dist = torch.mm(validation_embeddings, train_embeddings.T)
        yd, yi = dist.topk(self.knn_k, dim=1, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        predictions = torch.gather(candidates, 1, yi)
        predictions = predictions.narrow(1, 0, 1).clone().view(-1)

        self.accuracy(predictions, validation_labels)


def knn_predict(embeddings, train_embeddings, train_labels, k: int = 1):
    batch_size = embeddings.size(0)

    dist = torch.mm(embeddings, train_embeddings.T)
    yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
    candidates = train_labels.view(1, -1).expand(batch_size, -1)
    predictions = torch.gather(candidates, 1, yi)
    predictions = predictions.narrow(1, 0, 1).clone().view(-1)

    return predictions
