import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset


class CheckDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.zeros(10)))

    def val_dataloader(self):
        return [DataLoader(TensorDataset(torch.zeros(10)))]


class CheckModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


dm = CheckDataModule()
model = CheckModel()
trainer = Trainer()
trainer.fit(model, dm)
