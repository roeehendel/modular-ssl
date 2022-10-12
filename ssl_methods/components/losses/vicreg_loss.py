import torch
import torch.nn.functional as F

from ssl_methods.components.losses.siamese_pairwise_loss import SiamesePairwiseLoss
from utils.nn_utils.functional import off_diagonal, FullGatherLayer


class VICRegLoss(SiamesePairwiseLoss):
    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def _pair_loss(self, out1, out2):
        # print(out1.shape, out1.mean().item(), out1.std().item())
        # print(out2.shape, out2.mean().item(), out2.std().item())

        sim_loss = F.mse_loss(out1, out2)

        out1, out2 = self._dist_concat_and_center(out1), self._dist_concat_and_center(out2)

        std_loss = (self._std_loss(out1) + self._std_loss(out2)) / 2
        cov_loss = self._cov_loss(out1) + self._cov_loss(out2)

        # print(sim_loss.item(), std_loss.item(), cov_loss.item())

        loss = (self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss)

        return loss

    @staticmethod
    def _dist_concat_and_center(out):
        out = torch.cat(FullGatherLayer.apply(out), dim=0)
        out = out - out.mean(dim=0)
        return out

    @staticmethod
    def _std_loss(out):
        std = torch.sqrt(out.var(dim=0) + 0.0001)
        loss = torch.mean(F.relu(1 - std))
        return loss

    @staticmethod
    def _cov_loss(out):
        batch_size, embedding_dim = out.shape
        cov = (out.T @ out) / (batch_size - 1)
        loss = off_diagonal(cov).pow_(2).sum().div(embedding_dim)
        return loss
