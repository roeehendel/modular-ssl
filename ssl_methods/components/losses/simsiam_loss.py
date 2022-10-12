from torch import nn

from ssl_methods.components.losses.siamese_pairwise_loss import SiamesePairwiseLoss


class SimSiamLoss(SiamesePairwiseLoss):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim=1)

    def _pair_loss(self, out1, out2):
        z1, h1 = out1
        z2, h2 = out2
        z1 = z1.detach()
        return -self.criterion(h2, z1).mean()
